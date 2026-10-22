import os
import subprocess
from types import SimpleNamespace

import pytest

import triton.backends.ascend.compiler as compiler

pytestmark = pytest.mark.backend("none")


@pytest.fixture(autouse=True)
def mock_native_compiler_dependencies(monkeypatch):
    # Both flows now use the native A5 compiler path; no device/tool probes.
    monkeypatch.setattr(compiler, "NPUUtils", lambda: SimpleNamespace(has_device_limit=lambda: False))
    monkeypatch.setattr(compiler, "_check_bishengir_api_change", lambda: True)
    monkeypatch.setattr(compiler, "_npu_compiler_supports_option", lambda *args: True)
    monkeypatch.setattr(compiler, "_is_ascend_sanitizer_enabled", lambda: False)
    monkeypatch.setattr(compiler, "_is_debug_line_info_disabled", lambda: True)
    monkeypatch.setattr(compiler, "_enable_msdebug", lambda: False)
    monkeypatch.setattr(compiler, "_enable_print_ub_bits", lambda: False)
    monkeypatch.setattr(compiler, "_is_auto_map_parallel_blocks_enabled", lambda: True)


def _make_metadata(options):
    metadata = dict(options.__dict__)
    metadata["target"] = SimpleNamespace(arch=options.target_arch)
    metadata["hash"] = "unit-test-hash"
    return metadata


def _sample_linalg():
    return """
module attributes {mix_mode = "aiv", parallel_mode = "simd"} {
  func.func @vector_add_kernel(
    %arg0: memref<?xf32> {tt.tensor_kind = 0 : i32},
    %arg1: memref<?xf32> {tt.tensor_kind = 0 : i32},
    %arg2: memref<?xf32> {tt.tensor_kind = 1 : i32}) {
    return
  }
}
"""


def test_compile_flow_env_default_and_explicit_override(monkeypatch):
    monkeypatch.setenv("TRITON_ASCEND_COMPILE_FLOW", "ptoas")

    env_selected = compiler.NPUOptions(arch="Ascend950PR_9589")
    explicit_native = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="npuir")

    assert env_selected.compile_flow == "ptoas"
    assert explicit_native.compile_flow == "npuir"


def test_compile_flow_rejects_invalid_value():
    with pytest.raises(ValueError, match="invalid compile_flow"):
        compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="unknown")


def test_ptoas_compile_flow_stage_graph():
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend950PR_9589"))
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    stages = {}

    backend.add_stages(stages, options, language="ttir")

    assert list(stages.keys()) == ["ttir", "ttadapter", "mlirbc", "bcmlir", "ptovmi", "npubin"]


def test_ptoas_compile_flow_rejects_pure_simt():
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend950PR_9589"))
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas", compile_mode="simt_only")

    with pytest.raises(NotImplementedError, match="simt_only"):
        backend.add_stages({}, options, language="ttir")


def test_ptoas_compile_flow_accepts_mixed_aic_aiv_kernel():
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    metadata = {"mix_mode": "mix"}
    linalg = _sample_linalg().replace('mix_mode = "aiv"', 'mix_mode = "mix"')

    compiler._validate_ptoas_launcher_contract(linalg, metadata, options)


def test_linalg_to_ptoas_vmi_invokes_bishengir_emit(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    metadata = _make_metadata(options)
    commands = []

    monkeypatch.setattr(compiler, "_get_npucompiler_path",
                        lambda: ("/fake/bishengir-compile", os.environ.copy()))

    def fake_run(cmd, env=None, stdout=None, stderr=None, check=False, **kwargs):
        commands.append(cmd)
        output_path = cmd[-1]
        with open(output_path, "w") as f:
            f.write("module { pto.vmi.return }\n")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)

    result = compiler.linalg_to_bin_enable_npu_compile_910_95(
        _sample_linalg(), metadata, options, emit_ptoas_vmi=True)

    assert result == "module { pto.vmi.return }\n"
    assert metadata["kernel_name"] == "vector_add_kernel"
    assert metadata["name"] == "vector_add_kernel"
    assert metadata["mix_mode"] == "aiv"
    assert metadata["parallel_mode"] == "simd"
    assert metadata["tensor_kinds"] == [0, 0, 1]
    assert metadata["workspace_size"] == 0
    assert metadata["lock_num"] == 0
    assert metadata["lock_init_value"] == 0
    assert metadata["ptsm_cap_authorized"] is False
    assert metadata["auto_blockify_enabled"] is True
    assert commands[0][0] == "/fake/bishengir-compile"
    assert "--emit-ptoas-vmi" in commands[0]
    assert "--target=Ascend950PR_9589" in commands[0]
    assert "--enable-auto-blockify-loop" in commands[0]
    assert "--enable-hivm-compile=true" in commands[0]
    assert "--enable-triton-kernel-compile=true" in commands[0]


def test_linalg_to_ptoas_vmi_persists_debug_dump(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas", debug=True)
    metadata = _make_metadata(options)
    dumped = []

    monkeypatch.setattr(compiler, "_get_npucompiler_path",
                        lambda: ("/fake/bishengir-compile", os.environ.copy()))

    def fake_run(cmd, **kwargs):
        with open(cmd[-1], "w") as f:
            f.write("module { pto.vmi.return }\n")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    class FakeDumpManager:

        def _make_path(self, filename):
            return filename

        def put(self, data, filename, binary):
            dumped.append((data, filename, binary))

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    monkeypatch.setattr(compiler, "get_dump_manager", lambda key: FakeDumpManager())

    compiler.linalg_to_bin_enable_npu_compile_910_95(
        _sample_linalg(), metadata, options, emit_ptoas_vmi=True)

    assert ("module { pto.vmi.return }\n", "kernel.ptovmi.mlir", False) in dumped
    assert any(filename == "kernel.npuir.mlir" for _, filename, _ in dumped)


def test_ptoas_compile_flow_exports_disabled_auto_blockify(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    metadata = _make_metadata(options)
    metadata["has_auto_blockify_blacklist_op"] = True
    commands = []

    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", os.environ.copy()))

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        with open(cmd[-1], "w") as f:
            f.write("module { pto.vmi.return }\n")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)

    compiler.linalg_to_bin_enable_npu_compile_910_95(
        _sample_linalg(), metadata, options, emit_ptoas_vmi=True)

    assert metadata["auto_blockify_enabled"] is False
    assert "--enable-auto-blockify-loop" not in commands[0]


def test_ptoas_vmi_to_npubin_uses_direct_device_object(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    metadata = _make_metadata(options)
    commands = []

    monkeypatch.setattr(compiler, "_get_ptoas_path", lambda: ("/fake/ptoas", os.environ.copy()))

    def fake_run(cmd, env=None, stdout=None, stderr=None, check=False, **kwargs):
        commands.append(cmd)
        with open(cmd[-1], "wb") as f:
            f.write(b"device-object")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)

    result = compiler.ptoas_vmi_to_npubin("module { pto.vmi.return }\n", metadata, options)

    assert result == b"device-object"
    assert commands[0][:3] == ["/fake/ptoas", "--pto-backend=vpto", "--pto-arch=a5"]
    assert "--emit-device-object" in commands[0]
    assert len(commands) == 1


@pytest.fixture
def captured_compiles(monkeypatch):
    commands = []
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {"TOOL_ENV": "kept"}))

    def fake_run(cmd, env=None, **kwargs):
        assert env == {"TOOL_ENV": "kept"}
        commands.append(cmd)
        output = compiler.Path(cmd[-1])
        if output.suffix == ".mlir":
            output.write_text("module { pto.vmi.return }\n")
        else:
            output.with_suffix(".o").write_bytes(b"native-object")
        return subprocess.CompletedProcess(cmd, 0, b"UB size = 128 bits\n", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    return commands


@pytest.mark.parametrize("value", [None, False, True])
def test_native_and_bridge_forward_the_same_a5_options(value, captured_compiles, monkeypatch):
    # Enable every conditional native option, including controls absent from
    # the former bridge builder. False must be forwarded, not treated as None.
    options = compiler.NPUOptions(
        arch="Ascend950PR_9589", compile_flow="npuir", multibuffer=value,
        vf_fusion_mode="max-parallel", enable_preload=value,
        disable_tightly_coupled_buffer_reuse=True, enable_auto_bind_sub_block=value,
        enable_hivm_auto_cv_balance=value, sync_solver=value, unit_flag=value,
        inject_barrier_all=value, inject_block_all=value,
        limit_auto_multi_buffer_only_for_local_buffer=value,
        set_workspace_multibuffer=0, limit_auto_multi_buffer_of_local_buffer="no-limit",
        limit_auto_multi_buffer_buffer="L1", enable_mixed_cv=value, enable_vf_fusion=value,
        enable_flatten=value, enable_auto_vectorize_v2=value,
        auto_vectorize_v2_max_fused_ops_num=0, prevec_max_fused_ops_num=0,
        disable_auto_inject_block_sync=value, vf_merge_level=0,
        hfusion_enable_multiple_consumer_fusion=True, plan_memory_strategy="largest-first",
        bisheng_options="-custom-native-option",
    )
    monkeypatch.setattr(compiler, "NPUUtils", lambda: SimpleNamespace(
        has_device_limit=lambda: True, get_aicore_num=lambda: 1, get_aivector_core_num=lambda: 2))
    for name in ("_is_ascend_sanitizer_enabled", "_enable_msdebug", "_enable_print_ub_bits"):
        monkeypatch.setattr(compiler, name, lambda: True)
    monkeypatch.setattr(compiler, "_is_debug_line_info_disabled", lambda: False)
    linalg = _sample_linalg().replace('parallel_mode = "simd"', 'parallel_mode = "simd", bitcode = "/custom.bc"')
    native_metadata = _make_metadata(options)
    bridge_metadata = _make_metadata(options)

    native = compiler.linalg_to_bin_enable_npu_compile_910_95(linalg, native_metadata, options)
    bridge = compiler.linalg_to_bin_enable_npu_compile_910_95(
        linalg, bridge_metadata, options, emit_ptoas_vmi=True)

    assert native == b"native-object"
    assert bridge == "module { pto.vmi.return }\n"
    native_args = captured_compiles[0][2:-2]
    bridge_args = captured_compiles[1][2:-2]
    bridge_only = ["--emit-ptoas-vmi", "--enable-hivm-compile=true", "--mlir-disable-threading"]
    assert bridge_args == native_args + bridge_only
    for flag in ("--enable-hivm-unit-flag-sync", "--enable-hivm-graph-sync-solver", "--enable-preload"):
        assert any(arg.startswith(flag + "=") for arg in bridge_args) == (value is not None)
        if value is not None:
            assert f"{flag}={value}" in bridge_args
    for flag in ("--set-workspace-multibuffer", "--enable-vf-merge-level",
                 "--hfusion-max-fused-ops-in-auto-vectorize-v2", "--hfusion-max-fused-elementwise-ops"):
        assert f"{flag}=0" in bridge_args
    for flag in ("--custom-aic-number=1", "--custom-aiv-number=2", "--link-aicore-bitcode=/custom.bc",
                 "--append-bisheng-options=-custom-native-option", "--enable-lib-call-no-inline=false",
                 "--enable-sanitizer=true", "--enable-debug-info=true", "--enable-debug-variables=true",
                 "--enable-print-memory-allocated-size"):
        assert flag in bridge_args
    assert native_metadata["required_ub_bits"] == bridge_metadata["required_ub_bits"] == 128


@pytest.mark.parametrize("emit_vmi", [False, True])
def test_a5_preserves_native_resource_callbacks(emit_vmi, captured_compiles, monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589")
    metadata = _make_metadata(options)
    callbacks = []
    original_run = compiler.subprocess.run

    def fake_run(cmd, **kwargs):
        result = original_run(cmd, **kwargs)
        compiler.Path(cmd[-1]).parent.joinpath("libkernel.so").touch()
        return result

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    monkeypatch.setattr(compiler.ctypes, "CDLL", lambda path: object())

    def callback(lib, suffix, result, key):
        callbacks.append(key)
        result[key] = 17

    monkeypatch.setattr(compiler, "__get_metadata_attr_by_callback", callback)
    compiler.linalg_to_bin_enable_npu_compile_910_95(
        _sample_linalg(), metadata, options, emit_ptoas_vmi=emit_vmi)

    expected_callbacks = ["bs_task_type", "workspace_size", "sync_block_lock_layout", "lock_init_val"]
    assert callbacks == ([] if emit_vmi else expected_callbacks)
    assert metadata["workspace_size"] == (0 if emit_vmi else 17)


@pytest.mark.parametrize("linalg", [
    _sample_linalg().replace('mix_mode = "aiv"', 'mix_mode = "aic"'),
    _sample_linalg().replace("return", "hivm.sync_block_lock\n    return"),
    _sample_linalg().replace("return", "hivm.sync_block_lock_unordered\n    return"),
])
def test_shared_a5_path_keeps_bridge_launcher_guards(linalg, captured_compiles):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow="ptoas")
    with pytest.raises(NotImplementedError):
        compiler.linalg_to_bin_enable_npu_compile_910_95(
            linalg, _make_metadata(options), options, emit_ptoas_vmi=True)
    assert not captured_compiles


@pytest.mark.parametrize("flow", ["npuir", "ptoas"])
def test_stage_graph_calls_shared_a5_compiler(flow, monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589", compile_flow=flow)
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch=options.target_arch))
    calls = []

    def compile_a5(src, metadata, opt, *, emit_ptoas_vmi=False):
        calls.append((src, metadata, opt, emit_ptoas_vmi))
        return "vmi" if emit_ptoas_vmi else b"native-object"

    monkeypatch.setattr(compiler, "linalg_to_bin_enable_npu_compile_910_95", compile_a5)
    monkeypatch.setattr(compiler, "rewrite_debug_line", lambda artifact, **kwargs: artifact)
    stages = {}
    backend.add_stages(stages, options, language="ttir")
    assert list(stages)[:4] == ["ttir", "ttadapter", "mlirbc", "bcmlir"]
    metadata = {}
    stage_name = "ptovmi" if flow == "ptoas" else "npubin"
    assert stages[stage_name]("linalg", metadata) == ("vmi" if flow == "ptoas" else b"native-object")
    assert calls == [("linalg", metadata, options, flow == "ptoas")]


@pytest.mark.parametrize("emit_vmi", [False, True])
def test_shared_a5_path_propagates_compiler_failure(emit_vmi, monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589")
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))
    error = subprocess.CalledProcessError(1, "bishengir-compile", stderr=b"compile failed")

    def fail_compile(*args, **kwargs):
        raise error

    monkeypatch.setattr(compiler.subprocess, "run", fail_compile)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        compiler.linalg_to_bin_enable_npu_compile_910_95(
            _sample_linalg(), _make_metadata(options), options, emit_ptoas_vmi=emit_vmi)
    assert raised.value is error


def test_shared_a5_path_requires_vmi_output(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9589")
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))
    monkeypatch.setattr(compiler.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, b"", b""))
    with pytest.raises(FileNotFoundError, match="Expected PTOAS VMI output"):
        compiler.linalg_to_bin_enable_npu_compile_910_95(
            _sample_linalg(), _make_metadata(options), options, emit_ptoas_vmi=True)
