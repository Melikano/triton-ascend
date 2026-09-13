import os
import subprocess
from types import SimpleNamespace

import pytest

import triton.backends.ascend.compiler as compiler


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

    env_selected = compiler.NPUOptions(arch="Ascend950PR_9599")
    explicit_native = compiler.NPUOptions(arch="Ascend950PR_9599", compile_flow="npuir")

    assert env_selected.compile_flow == "ptoas"
    assert explicit_native.compile_flow == "npuir"


def test_compile_flow_rejects_invalid_value():
    with pytest.raises(ValueError, match="invalid compile_flow"):
        compiler.NPUOptions(arch="Ascend950PR_9599", compile_flow="unknown")


def test_ptoas_compile_flow_stage_graph():
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend950PR_9599"))
    options = compiler.NPUOptions(arch="Ascend950PR_9599", compile_flow="ptoas")
    stages = {}

    backend.add_stages(stages, options, language="ttir")

    assert list(stages.keys()) == ["ttir", "ttadapter", "mlirbc", "bcmlir", "ptovmi", "npubin"]


def test_ptoas_compile_flow_rejects_pure_simt():
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend950PR_9599"))
    options = compiler.NPUOptions(arch="Ascend950PR_9599", compile_flow="ptoas", compile_mode="simt_only")

    with pytest.raises(NotImplementedError, match="simt_only"):
        backend.add_stages({}, options, language="ttir")


def test_linalg_to_ptoas_vmi_invokes_bishengir_emit(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9599", compile_flow="ptoas")
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

    result = compiler.linalg_to_ptoas_vmi(_sample_linalg(), metadata, options)

    assert result == "module { pto.vmi.return }\n"
    assert metadata["kernel_name"] == "vector_add_kernel"
    assert metadata["name"] == "vector_add_kernel"
    assert metadata["mix_mode"] == "aiv"
    assert metadata["parallel_mode"] == "simd"
    assert metadata["tensor_kinds"] == [0, 0, 1]
    assert metadata["workspace_size"] == 0
    assert metadata["lock_num"] == 0
    assert metadata["lock_init_value"] == 0
    assert metadata["auto_blockify_enabled"] is True
    assert commands[0][0] == "/fake/bishengir-compile"
    assert "--emit-ptoas-vmi" in commands[0]
    assert "--target=Ascend950PR_9599" in commands[0]
    assert "--enable-auto-blockify-loop" in commands[0]
    assert "--enable-hivm-compile=true" in commands[0]
    assert "--enable-triton-kernel-compile=true" in commands[0]


def test_ptoas_compile_flow_exports_disabled_auto_blockify(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9599", compile_flow="ptoas")
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

    compiler.linalg_to_ptoas_vmi(_sample_linalg(), metadata, options)

    assert metadata["auto_blockify_enabled"] is False
    assert "--enable-auto-blockify-loop" not in commands[0]


def test_ptoas_vmi_to_npubin_uses_direct_device_object(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend950PR_9599", compile_flow="ptoas")
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
