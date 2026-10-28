"""CPU-only HOST metadata tests; no NPU or simulator is launched."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import triton.backends.ascend.compiler as compiler
from triton.backends.ascend import ptoas_host

pytestmark = pytest.mark.backend("none")


@pytest.fixture
def native_callback_library(tmp_path):
    source = tmp_path / "callbacks.c"
    source.write_text("""long long vector_add_kernel_infer_workspace_shape_function(void) { return 16; }
long long vector_add_kernel_infer_task_type_function(void) { return 21; }
""")
    library = tmp_path / "callbacks.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


def test_bridge_reads_host_callbacks_before_returning_vmi(monkeypatch, native_callback_library):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    metadata = dict(options.__dict__, target=SimpleNamespace(arch=options.target_arch), hash="host-test")
    monkeypatch.setattr(compiler, "NPUUtils", lambda: SimpleNamespace(has_device_limit=lambda: False))
    monkeypatch.setattr(compiler, "_check_bishengir_api_change", lambda: True)
    monkeypatch.setattr(compiler, "_npu_compiler_supports_option", lambda *args: True)
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", os.environ.copy()))
    monkeypatch.setattr(compiler, "_require_ptoas_host_export", lambda *args: None, raising=False)
    monkeypatch.setattr(compiler, "_is_ascend_sanitizer_enabled", lambda: False)
    monkeypatch.setattr(compiler, "_is_debug_line_info_disabled", lambda: True)
    monkeypatch.setattr(compiler, "_enable_msdebug", lambda: False)
    monkeypatch.setattr(compiler, "_enable_print_ub_bits", lambda: False)

    def device_compile(cmd, **kwargs):
        output = Path(cmd[-1])
        output.write_text("module { pto.vmi.return }\n")
        Path(str(output) + ".host.mlir").write_text("module {}\n")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def host_compile(host_path, compiler_path, env, kernel_name):
        library = Path(host_path).with_name("libkernel.so")
        shutil.copyfile(native_callback_library, library)
        return library, {"workspace_size": (kernel_name + "_infer_workspace_shape_function", 64),
                         "bs_task_type": (kernel_name + "_infer_task_type_function", 64)}

    monkeypatch.setattr(compiler.subprocess, "run", device_compile)
    monkeypatch.setattr(compiler, "_compile_ptoas_host_callbacks", host_compile, raising=False)
    linalg = '''module attributes {mix_mode = "aiv", parallel_mode = "simd"} {
      func.func @vector_add_kernel() { return }
    }'''
    result = compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True)
    assert result == "module { pto.vmi.return }\n"
    assert metadata["workspace_size"] == 16
    assert metadata["bs_task_type"] == 21
    assert not any(str(value).endswith((".so", ".host.mlir")) for value in metadata.values())


@pytest.fixture
def host_tools():
    directory = os.getenv("TRITON_TEST_HOST_TOOL_DIR")
    cc = os.getenv("TRITON_TEST_HOST_CC")
    if not directory or not cc:
        pytest.skip("Set TRITON_TEST_HOST_TOOL_DIR and TRITON_TEST_HOST_CC for real CPU tool checks")
    directory = Path(directory)
    for name in ("bishengir-opt", "mlir-translate"):
        assert (directory / name).is_file(), name
    env = os.environ.copy()
    env["TRITON_PTOAS_HOST_CC"] = cc
    return str(directory / "bishengir-compile"), env


def _host_module(value=16, symbol="vector_add_kernel_infer_workspace_shape_function", result="index", task_result="i64"):
    return f'''module attributes {{hacc.target = #hacc.target<"Ascend950PR_9589">,
                                  llvm.target_triple = "aicore-unknown-linux"}} {{
      func.func private @vector_add_kernel() attributes {{
        hacc.infer_workspace_shape_function = #hacc.infer_workspace_shape_function<@{symbol}>}}
      func.func private @helper(%x: {result}) -> {result} attributes {{hacc.function_kind = #hacc.function_kind<HOST>}} {{
        %offset = arith.constant {value} : {result}
        %sum = arith.addi %x, %offset : {result}
        return %sum : {result}
      }}
      func.func @{symbol}() -> {result} attributes {{
        hacc.function_kind = #hacc.function_kind<HOST>,
        hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}} {{
        %zero = arith.constant 0 : {result}
        %size = func.call @helper(%zero) : ({result}) -> {result}
        return %size : {result}
      }}
      func.func @vector_add_kernel_infer_task_type_function() -> {task_result} attributes {{
        hacc.function_kind = #hacc.function_kind<HOST>}} {{
        %task = arith.constant 21 : {task_result}
        return %task : {task_result}
      }}
    }}'''


@pytest.mark.parametrize("value", [0, 16, 64, 2**40])
@pytest.mark.parametrize("symbol", ["vector_add_kernel_infer_workspace_shape_function", "bound_workspace"])
def test_real_host_lowering_and_cpu_callbacks(value, symbol, tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    host.write_text(_host_module(value, symbol))
    library, symbols = ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    metadata = {"kernel_name": "vector_add_kernel"}
    compiler._read_metadata_callbacks(library, metadata, symbols=symbols)
    assert metadata["workspace_size"] == value
    assert metadata["bs_task_type"] == 21
    llvm_ir = (tmp_path / "kernel.host.ll").read_text()
    assert 'target triple = "aicore' not in llvm_ir
    assert "ret i64" in llvm_ir


@pytest.mark.parametrize("result", ["i32", "f32"])
def test_real_host_rejects_non_native_return_abi(result, tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    if result == "f32":
        host.write_text('''module { func.func @vector_add_kernel_infer_workspace_shape_function() -> f32 {
          %v = arith.constant 16.0 : f32
          return %v : f32
        }}''')
    else:
        host.write_text(_host_module(result=result))
    with pytest.raises(RuntimeError, match=r"C-ABI \(\) -> i64"):
        ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    assert not (tmp_path / "libkernel.so").exists()


def test_real_host_rejects_callback_arguments(tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    host.write_text('''module { func.func @vector_add_kernel_infer_workspace_shape_function(%n: index) -> index {
      return %n : index
    }}''')
    with pytest.raises(RuntimeError, match=r"C-ABI \(\) -> i64"):
        ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")


def test_real_callback_free_host_library(tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    host.write_text("module {}")
    library, symbols = ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    assert symbols == {}
    assert ctypes.CDLL(str(library)) is not None


def test_real_native_i8_task_type_abi(tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    host.write_text(_host_module(task_result="i8"))
    library, symbols = ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    assert symbols["bs_task_type"] == ("vector_add_kernel_infer_task_type_function", 8)
    metadata = {"kernel_name": "vector_add_kernel"}
    compiler._read_metadata_callbacks(library, metadata, symbols=symbols)
    assert metadata["bs_task_type"] == 21
    assert metadata["workspace_size"] == 16


def test_real_cpu_index_abi_overrides_device_layout(tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    text = _host_module(2**40)
    text = text.replace('llvm.target_triple = "aicore-unknown-linux"',
                        'llvm.target_triple = "aicore-unknown-linux", '
                        'dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>')
    host.write_text(text)
    library, symbols = ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    metadata = {"kernel_name": "vector_add_kernel"}
    compiler._read_metadata_callbacks(library, metadata, symbols=symbols)
    assert metadata["workspace_size"] == 2**40


def test_real_affine_and_rounded_shape_callback(tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    text = _host_module().replace("%sum = arith.addi %x, %offset : index",
                                 "%sum = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%x, %offset)")
    text = text.replace("return %sum : index", "%divisor = arith.constant 3 : index\n"
                        "%rounded = arith.ceildivui %sum, %divisor : index\nreturn %rounded : index")
    host.write_text(text)
    library, symbols = ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    metadata = {"kernel_name": "vector_add_kernel"}
    compiler._read_metadata_callbacks(library, metadata, symbols=symbols)
    assert metadata["workspace_size"] == 6


def test_callback_library_load_and_missing_symbol_errors_propagate(tmp_path, native_callback_library):
    metadata = {"kernel_name": "vector_add_kernel"}
    with pytest.raises(OSError):
        compiler._read_metadata_callbacks(tmp_path / "missing.so", metadata, symbols={})
    with pytest.raises(RuntimeError, match="missing from"):
        compiler._read_metadata_callbacks(native_callback_library, metadata,
                                          symbols={"workspace_size": ("missing_callback", 64)})


def test_real_host_lowering_failure_is_not_zero_workspace(tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    host.write_text("not valid MLIR")
    with pytest.raises(subprocess.CalledProcessError):
        ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    assert not (tmp_path / "libkernel.so").exists()


def test_real_host_link_failure_propagates(tmp_path, host_tools):
    compiler_path, env = host_tools
    host = tmp_path / "kernel.ptovmi.mlir.host.mlir"
    host.write_text('''module {
      func.func private @missing_helper() -> index
      func.func @vector_add_kernel_infer_workspace_shape_function() -> index {
        %n = func.call @missing_helper() : () -> index
        return %n : index
      }
    }''')
    with pytest.raises(subprocess.CalledProcessError) as raised:
        ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    assert "-shared" in raised.value.cmd


def test_missing_host_compiler_is_reported(tmp_path, host_tools):
    compiler_path, env = host_tools
    env["TRITON_PTOAS_HOST_CC"] = str(tmp_path / "missing-compiler")
    with pytest.raises(RuntimeError, match="CPU LLVM compiler"):
        ptoas_host.compile_host_callbacks(tmp_path / "unused.host.mlir", compiler_path, env, "vector_add_kernel")


def test_missing_matching_tool_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="matching bishengir-opt"):
        ptoas_host.require_host_export(str(tmp_path / "bishengir-compile"), os.environ.copy())


def test_export_contract_rejects_old_tools(monkeypatch, tmp_path):
    opt = tmp_path / "bishengir-opt"
    opt.write_text("#!/bin/sh\nexit 0\n")
    opt.chmod(0o700)
    with pytest.raises(RuntimeError, match="automatic .host.mlir export contract"):
        ptoas_host.require_host_export(str(tmp_path / "bishengir-compile"), os.environ.copy())


@pytest.mark.parametrize("header", ["i32", "fastcc i64", "internal i64", "i64"])
def test_abi_validation_rejects_wrong_type_convention_visibility_or_args(header):
    args = "i64 %n" if header == "i64" else ""
    text = f"define {header} @k_infer_workspace_shape_function({args}) {{\nret i64 0\n}}"
    with pytest.raises(RuntimeError, match="exported C-ABI"):
        ptoas_host._callback_symbols("", text, "k")


@pytest.fixture
def bridge_device_stub(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    from triton.backends.compiler import GPUTarget
    metadata = dict(options.__dict__, target=GPUTarget("npu", options.target_arch, 32), hash="host-cache-test")
    monkeypatch.setattr(compiler, "NPUUtils", lambda: SimpleNamespace(has_device_limit=lambda: False))
    monkeypatch.setattr(compiler, "_check_bishengir_api_change", lambda: True)
    monkeypatch.setattr(compiler, "_npu_compiler_supports_option", lambda *args: True)
    monkeypatch.setattr(compiler, "_require_ptoas_host_export", lambda *args: None)
    for name in ("_is_ascend_sanitizer_enabled", "_enable_msdebug", "_enable_print_ub_bits"):
        monkeypatch.setattr(compiler, name, lambda: False)
    monkeypatch.setattr(compiler, "_is_debug_line_info_disabled", lambda: True)
    linalg = '''module attributes {mix_mode = "aiv", parallel_mode = "simd"} {
      func.func @vector_add_kernel() { return }
    }'''
    return options, metadata, linalg


@pytest.mark.parametrize("value", [0, 16, 64])
def test_bridge_real_host_metadata_survives_compiled_kernel_cache(
        value, host_tools, bridge_device_stub, monkeypatch, tmp_path):
    compiler_path, env = host_tools
    options, metadata, linalg = bridge_device_stub
    metadata["workspace_size"] = 999  # A real zero callback must overwrite stale metadata.
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: (compiler_path, env))
    original_run = subprocess.run
    outputs = []

    def fake_device_only(cmd, **kwargs):
        if cmd[0] != compiler_path:
            return original_run(cmd, **kwargs)
        output = Path(cmd[-1])
        outputs.append(output)
        output.write_text("module {}\n")
        Path(str(output) + ".host.mlir").write_text(_host_module(value))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_device_only)
    vmi = compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True)
    assert metadata["workspace_size"] == value
    assert metadata["bs_task_type"] == 21
    assert all(not output.exists() for output in outputs)  # Temporary artifacts have already gone.

    from triton import knobs
    from triton.runtime.cache import FileCacheManager
    import triton.compiler.compiler as core
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path / "cache"))
    cache = FileCacheManager(f"workspace-{value}")
    group = {"kernel.json": cache.put(json.dumps(metadata, default=vars), "kernel.json", binary=False),
             "kernel.npubin": cache.put(b"mock-device-object", "kernel.npubin"),
             "kernel.ptovmi": cache.put(vmi, "kernel.ptovmi", binary=False)}
    cache.put_group("kernel.json", group)
    monkeypatch.setattr(core, "make_backend", lambda target: compiler.AscendBackend(target))
    fresh = core.CompiledKernel(None, group, "host-cache-test")
    monkeypatch.setattr(compiler, "_compile_ptoas_host_callbacks",
                        lambda *args: pytest.fail("Cached metadata must not recompile host callbacks"))
    cached = core.CompiledKernel(None, cache.get_group("kernel.json"), "host-cache-test")
    assert fresh.metadata == cached.metadata
    assert cached.metadata.workspace_size == value
    assert cached.metadata.bs_task_type == 21
    assert cached.module is None and cached.function is None  # No device initialization.
    assert not any(str(v).endswith((".host.mlir", ".so")) for v in cached.metadata)


@pytest.mark.parametrize("value", [0, 16])
def test_actual_triton_compile_cache_hit_needs_no_host_artifacts(
        value, host_tools, bridge_device_stub, monkeypatch, tmp_path):
    from triton import knobs
    import triton.compiler.compiler as core
    import triton.runtime.cache as cache_module
    compiler_path, env = host_tools
    options, initial, linalg = bridge_device_stub
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: (compiler_path, env))
    original_run = subprocess.run
    device_calls = []

    def fake_device_only(cmd, **kwargs):
        if cmd[0] != compiler_path:
            return original_run(cmd, **kwargs)
        device_calls.append(cmd)
        output = Path(cmd[-1])
        output.write_text("module {}\n")
        Path(str(output) + ".host.mlir").write_text(_host_module(value, task_result="i8"))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_device_only)
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path / "cache"))
    monkeypatch.setattr(knobs.compilation, "always_compile", False)
    # Avoid walking/importing optional backend plugins (including torch_npu)
    # solely to construct the global frontend-version token in this CPU test.
    # The actual compile cache, serialization and cache-hit path remain real.
    monkeypatch.setattr(cache_module, "triton_key", lambda: "host-metadata-source-under-test")
    monkeypatch.setattr(compiler, "get_cann_version_file_hash", lambda: "cpu-callback-test-cann-version")
    backend = compiler.AscendBackend(initial["target"])

    def stages(out, parsed_options, language):
        out["ptovmi"] = lambda src, meta: compiler.linalg_to_bin_enable_npu_compile_910_95(
            src, meta, parsed_options, emit_ptoas_vmi=True)
        out["npubin"] = lambda src, meta: b"mock-device-object"

    monkeypatch.setattr(backend, "add_stages", stages)
    monkeypatch.setattr(core, "make_backend", lambda target: backend)
    fn = SimpleNamespace(__name__="vector_add_kernel", cache_key=f"host-workspace-{value}")
    source = core.ASTSource(fn, {})
    source.ext = "ptovmi"
    source.make_ir = lambda *args, **kwargs: linalg
    fresh = core.compile(source, target=initial["target"], options={"compile_flow": "ptoas"})
    assert fresh.metadata.workspace_size == value
    assert fresh.metadata.bs_task_type == 21
    assert len(device_calls) == 1
    monkeypatch.setattr(compiler, "_compile_ptoas_host_callbacks",
                        lambda *a: pytest.fail("A cache hit must not compile HOST metadata"))
    cached = core.compile(source, target=initial["target"], options={"compile_flow": "ptoas"})
    assert cached.metadata == fresh.metadata
    assert len(device_calls) == 1
    assert cached.module is None and cached.function is None


def test_failed_device_compilation_never_consumes_its_sidecar(bridge_device_stub, monkeypatch):
    options, metadata, linalg = bridge_device_stub
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))
    error = subprocess.CalledProcessError(1, "device", stderr=b"device compilation failed")

    def fail_device(cmd, **kwargs):
        Path(str(cmd[-1]) + ".host.mlir").write_text(_host_module())
        raise error

    monkeypatch.setattr(compiler.subprocess, "run", fail_device)
    monkeypatch.setattr(compiler, "_compile_ptoas_host_callbacks", lambda *a: pytest.fail("Sidecar from failed compile"))
    with pytest.raises(subprocess.CalledProcessError) as raised:
        compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True)
    assert raised.value is error


def test_successful_callback_free_device_needs_no_host_compiler(bridge_device_stub, monkeypatch):
    options, metadata, linalg = bridge_device_stub
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))

    def device(cmd, **kwargs):
        Path(cmd[-1]).write_text("module {}\n")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", device)
    monkeypatch.setattr(compiler, "_compile_ptoas_host_callbacks", lambda *a: pytest.fail("No HOST sidecar"))
    assert compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True) == "module {}\n"
    assert metadata["workspace_size"] == 0


def test_missing_known_host_sidecar_is_an_error(bridge_device_stub, monkeypatch):
    options, metadata, linalg = bridge_device_stub
    linalg = linalg.rsplit("}", 1)[0] + '''
      func.func @workspace_callback() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
        %n = arith.constant 16 : index
        return %n : index
      }
    }'''
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))

    def device(cmd, **kwargs):
        Path(cmd[-1]).write_text("module {}\n")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", device)
    with pytest.raises(FileNotFoundError, match="automatic HOST sidecar"):
        compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True)


@pytest.mark.parametrize("value", [-1, 1, 2**32])
def test_real_host_lock_metadata_keeps_launcher_restriction(value, host_tools, bridge_device_stub, monkeypatch):
    compiler_path, env = host_tools
    options, metadata, linalg = bridge_device_stub
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: (compiler_path, env))
    original_run = subprocess.run

    def fake_device_only(cmd, **kwargs):
        if cmd[0] != compiler_path:
            return original_run(cmd, **kwargs)
        output = Path(cmd[-1])
        output.write_text("module {}\n")
        Path(str(output) + ".host.mlir").write_text(f'''module {{
          func.func @vector_add_kernel_infer_sync_block_lock_num_function() -> i64 {{
            %v = arith.constant {value} : i64
            return %v : i64
          }}
        }}''')
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_device_only)
    with pytest.raises(NotImplementedError, match="sync-block-lock"):
        compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True)


@pytest.mark.parametrize("size", [0, 16, 64])
def test_launcher_workspace_allocation_and_argument(size, bridge_device_stub, monkeypatch):
    from triton.backends.ascend import driver
    options, metadata, _ = bridge_device_stub
    compiler._set_ptoas_launcher_defaults(metadata)
    metadata.update(name="vector_add_kernel", kernel_name="vector_add_kernel", mix_mode="aiv",
                    tensor_kinds=[], workspace_size=size)
    monkeypatch.setattr(driver, "NPUUtils", lambda: SimpleNamespace(
        has_device_limit=lambda: False, get_aivector_core_num=lambda: 2, get_aicore_num=lambda: 1,
        get_so_path=lambda: "/fake/npu-utils.so"))
    source = driver.make_launcher({}, {}, SimpleNamespace(**metadata))
    assert "uint32_t blockNum4Workspace = gridX * gridY * gridZ;" in source
    if size:
        assert f"uint64_t totalWorkSpaceSize = (uint64_t){size} * blockNum4Workspace;" in source
        assert "static_cast<void*>(workspace_addr_ptr)," in source
    else:
        assert "uint64_t totalWorkSpaceSize =" not in source
        assert "nullptr," in source


@pytest.fixture
def mainline_tools():
    directory = os.getenv("TRITON_TEST_MAINLINE_TOOL_DIR")
    cc = os.getenv("TRITON_TEST_HOST_CC")
    if not directory or not cc:
        pytest.skip("Set TRITON_TEST_MAINLINE_TOOL_DIR and TRITON_TEST_HOST_CC for actual mainline compilation")
    compiler_path = str(Path(directory) / "bishengir-compile")
    env = os.environ.copy()
    env["TRITON_PTOAS_HOST_CC"] = cc
    ptoas_host.require_host_export(compiler_path, env)
    return compiler_path, env


@pytest.mark.parametrize("value", [0, 16, 64])
def test_actual_mainline_automatic_sidecar_and_cpu_callback(value, mainline_tools, tmp_path):
    compiler_path, env = mainline_tools
    source = Path(__file__).with_name("fixtures") / "ptoas_host_metadata.mlir"
    fixture = tmp_path / "input.mlir"
    fixture.write_text(source.read_text().replace("arith.constant 16 : index", f"arith.constant {value} : index"))
    device = tmp_path / "kernel.ptovmi.mlir"
    subprocess.run([compiler_path, str(fixture), "--emit-ptoas-vmi", "--target=Ascend950PR_9589",
                    "--enable-hfusion-compile=false", "--enable-hivm-compile=true",
                    "--enable-triton-kernel-compile=true", "-o", str(device)],
                   env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert "hacc.function_kind<HOST>" not in device.read_text()
    assert "host_workspace_helper" not in device.read_text()
    host = Path(str(device) + ".host.mlir")
    assert host.is_file()
    assert "host_workspace_helper" in host.read_text()
    library, symbols = ptoas_host.compile_host_callbacks(host, compiler_path, env, "vector_add_kernel")
    metadata = {"kernel_name": "vector_add_kernel"}
    compiler._read_metadata_callbacks(library, metadata, symbols=symbols)
    assert metadata["workspace_size"] == value


@pytest.mark.parametrize("value", [0, 16, 64])
def test_actual_mainline_backend_returns_vmi_with_host_metadata(value, mainline_tools, bridge_device_stub, monkeypatch):
    compiler_path, env = mainline_tools
    options, metadata, _ = bridge_device_stub
    source = Path(__file__).with_name("fixtures") / "ptoas_host_metadata.mlir"
    linalg = source.read_text().replace("arith.constant 16 : index", f"arith.constant {value} : index")
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: (compiler_path, env))
    monkeypatch.setattr(compiler, "_require_ptoas_host_export", ptoas_host.require_host_export)
    result = compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True)
    assert isinstance(result, str)
    assert "hacc.function_kind<HOST>" not in result
    assert metadata["workspace_size"] == value


def test_actual_mainline_callback_free_compilation(mainline_tools, bridge_device_stub, monkeypatch):
    compiler_path, env = mainline_tools
    options, metadata, _ = bridge_device_stub
    linalg = '''module attributes {mix_mode = "aiv", parallel_mode = "simd"} {
      func.func @vector_add_kernel() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
        return
      }
    }'''
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: (compiler_path, env))
    monkeypatch.setattr(compiler, "_require_ptoas_host_export", ptoas_host.require_host_export)
    monkeypatch.setattr(compiler, "_compile_ptoas_host_callbacks", lambda *a: pytest.fail("Callback-free kernel"))
    result = compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, metadata, options, emit_ptoas_vmi=True)
    assert isinstance(result, str)
    assert metadata["workspace_size"] == 0
