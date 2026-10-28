"""Compile mainline's automatic HOST MLIR sidecar into native CPU callbacks.

Only values returned by the callbacks belong in kernel metadata. All paths in
this module are temporary compilation artifacts, never launcher/cache inputs.
"""
import ctypes
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


METADATA_CALLBACKS = (
    ("_infer_task_type_function", "bs_task_type"),
    ("_infer_workspace_shape_function", "workspace_size"),
    ("_infer_sync_block_lock_num_function", "sync_block_lock_layout"),
    ("_infer_sync_block_lock_init_function", "lock_init_val"),
)

_SYMBOL = r'(?:"(?:[^"\\]|\\.)*"|[-a-zA-Z$._][-a-zA-Z$._0-9]*)'


def _matching_tool(compiler_path, name, env):
    compiler = shutil.which(str(compiler_path), path=env.get("PATH", ""))
    compiler = Path(compiler or compiler_path).resolve()
    tool = compiler.parent / name
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise RuntimeError(f"PTOAS HOST metadata requires matching {name} alongside {compiler}; missing {tool}")
    return str(tool)


def require_host_export(compiler_path, env):
    # A successful old compiler invocation is not evidence that an absent
    # sidecar means 'no callbacks'. Require the matching mainline separation
    # capability, even for callback-free kernels.
    tool = _matching_tool(compiler_path, "bishengir-opt", env)
    result = subprocess.run([tool, "--help"], env=env, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if b"--separate-ptoas-host-module" not in result.stdout:
        raise RuntimeError("PTOAS HOST metadata requires the mainline automatic .host.mlir export contract; "
                           f"{tool} lacks separate-ptoas-host-module. Use matching mainline compiler tools.")


def _run(command, env, **kwargs):
    return subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def _cpu_target(cc, env):
    result = _run([cc, "-x", "c", "-S", "-emit-llvm", "-o", "-", "-"], env, input=b"")
    text = result.stdout.decode()
    layout = re.search(r'^target datalayout = "([^"]+)"', text, re.M)
    triple = re.search(r'^target triple = "([^"]+)"', text, re.M)
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    if ctypes.sizeof(ctypes.c_void_p) != 8 or not layout or not triple or triple[1].split("-")[0] != machine:
        raise RuntimeError(f"PTOAS metadata compiler {cc} must emit LLVM IR for this 64-bit CPU ({machine})")
    return layout[1], triple[1]


def _callback_symbols(lowered_mlir, llvm_ir, kernel_name):
    # Read symbol bindings/roles, never constant return values. Both texts are
    # canonical output from matching MLIR tools, not arbitrary source snippets.
    headers = {}
    for line in lowered_mlir.splitlines():
        match = re.match(r'\s*llvm\.func\s+@(' + _SYMBOL + r')\(', line)
        if match:
            headers[match[1].strip('"')] = line
    definitions = {}
    pattern = r'^define\s+([^@\n]+)@(' + _SYMBOL + r')\(([^)]*)\)'
    for match in re.finditer(pattern, llvm_ir, re.M):
        definitions[match[2].strip('"')] = (match[1].split(), match[3].strip())

    symbols = {}
    for suffix, key in METADATA_CALLBACKS:
        kind = suffix[1:]
        bound = set()
        for name, header in headers.items():
            if name == kernel_name or name.startswith(kernel_name + "_"):
                binding = re.search(r'\bhacc\.' + kind + r'\s*=\s*#hacc\.' + kind + r'<@(' + _SYMBOL + r')>', header)
                if binding:
                    bound.add(binding[1].strip('"'))
        if len(bound) > 1:
            raise RuntimeError(f"Ambiguous HOST callback bindings for {key}: {sorted(bound)}")
        native_name = kernel_name + suffix
        if bound:
            symbol = next(iter(bound))
        elif native_name in definitions:
            symbol = native_name
        else:
            typed = [name for name, header in headers.items()
                     if re.search(r'hacc\.host_func_type\s*=\s*#hacc\.host_func_type<' + kind + r'>', header)]
            if len(typed) > 1:
                raise RuntimeError(f"Ambiguous HOST callback roles for {key}: {typed}")
            if not typed:
                continue
            symbol = typed[0]

        prototype = definitions.get(symbol)
        return_types = {"i8", "i64"} if key == "bs_task_type" else {"i64"}
        allowed = {"dso_local", "dso_preemptable", "noundef", "signext", "zeroext", "ccc"} | return_types
        if (not prototype or prototype[1] or not prototype[0] or prototype[0][-1] not in return_types
                or any(token not in allowed for token in prototype[0])):
            expected = "i8/i64" if key == "bs_task_type" else "i64"
            raise RuntimeError(f"HOST metadata callback {symbol!r} must be an exported C-ABI () -> {expected} definition")
        symbols[key] = (symbol, int(prototype[0][-1][1:]))

    # Mainline retains device declarations in the host sidecar, but executable
    # device functions must never enter CPU code generation.
    for name in definitions:
        header = headers.get(name, "")
        if re.search(r'hacc\.function_kind<DEVICE|hivm\.func_core_type', header):
            raise RuntimeError(f"Device definition {name!r} is not valid in a HOST metadata module")
    return symbols


def compile_host_callbacks(host_path, compiler_path, env, kernel_name):
    host_path = Path(host_path)
    opt = _matching_tool(compiler_path, "bishengir-opt", env)
    translate = _matching_tool(compiler_path, "mlir-translate", env)
    explicit_cc = env.get("TRITON_PTOAS_HOST_CC")
    cc = shutil.which(explicit_cc or "bisheng", path=env.get("PATH", ""))
    if not explicit_cc and not cc:
        cc = shutil.which("clang", path=env.get("PATH", ""))
    if not cc:
        raise RuntimeError("PTOAS HOST metadata needs a CPU LLVM compiler; set TRITON_PTOAS_HOST_CC to bisheng/clang")
    layout, triple = _cpu_target(cc, env)
    lowered = host_path.with_name("kernel.host.llvm.mlir")
    llvm_path = host_path.with_name("kernel.host.ll")
    library = host_path.with_name("libkernel.so")
    _run([opt, str(host_path), f'--set-llvm-module-datalayout=data-layout="{layout}"',
          "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf", "--arith-expand", "--convert-math-to-llvm",
          "--finalize-memref-to-llvm=index-bitwidth=64", "--convert-index-to-llvm=index-bitwidth=64",
          "--convert-arith-to-llvm", "--convert-cf-to-llvm", "--convert-func-to-llvm=index-bitwidth=64",
          "--reconcile-unrealized-casts", "-o", str(lowered)], env)
    _run([translate, "--mlir-to-llvmir", "--allow-unregistered-dialect", str(lowered), "-o", str(llvm_path)], env)
    llvm_ir = llvm_path.read_text()
    symbols = _callback_symbols(lowered.read_text(), llvm_ir, kernel_name)
    # The host clone retains NPU module attributes. CPU lowering explicitly uses
    # 64-bit indices and the native layout; CPU codegen receives only its native
    # target, never an inherited NPU triple/layout.
    llvm_ir = re.sub(r'^target (?:triple|datalayout) = .*\n?', "", llvm_ir, flags=re.M)
    llvm_path.write_text(f'target datalayout = "{layout}"\ntarget triple = "{triple}"\n' + llvm_ir)
    _run([cc, "-shared", "-fPIC", "-Wl,-z,defs", str(llvm_path), "-o", str(library)], env)
    if not library.is_file():
        raise FileNotFoundError(f"HOST callback library was not produced: {library}")
    return library, symbols
