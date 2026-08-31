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

    env_selected = compiler.NPUOptions(arch="Ascend910_9589")
    explicit_native = compiler.NPUOptions(arch="Ascend910_9589", compile_flow="npuir")

    assert env_selected.compile_flow == "ptoas"
    assert explicit_native.compile_flow == "npuir"


def test_compile_flow_rejects_invalid_value():
    with pytest.raises(ValueError, match="invalid compile_flow"):
        compiler.NPUOptions(arch="Ascend910_9589", compile_flow="unknown")


def test_ptoas_compile_flow_stage_graph():
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend910_9589"))
    options = compiler.NPUOptions(arch="Ascend910_9589", compile_flow="ptoas")
    stages = {}

    backend.add_stages(stages, options, language="ttir")

    assert list(stages.keys()) == ["ttir", "ttadapter", "mlirbc", "bcmlir", "ptovmi", "npubin"]


def test_ptoas_compile_flow_rejects_pure_simt():
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend910_9589"))
    options = compiler.NPUOptions(arch="Ascend910_9589", compile_flow="ptoas", compile_mode="simt_only")

    with pytest.raises(NotImplementedError, match="simt_only"):
        backend.add_stages({}, options, language="ttir")


def test_linalg_to_ptoas_vmi_invokes_bishengir_emit(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend910_9589", compile_flow="ptoas")
    metadata = _make_metadata(options)
    commands = []

    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", os.environ.copy()))

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
    assert commands[0][0] == "/fake/bishengir-compile"
    assert "--emit-ptoas-vmi" in commands[0]
    assert "--target=Ascend910_9589" in commands[0]
    assert "--enable-hivm-compile=true" in commands[0]
    assert "--enable-triton-kernel-compile=true" in commands[0]


def test_ptoas_vmi_to_npubin_extracts_and_links_raw_device_object(monkeypatch):
    options = compiler.NPUOptions(arch="Ascend910_9589", compile_flow="ptoas")
    metadata = _make_metadata(options)
    commands = []

    monkeypatch.setattr(compiler, "_get_ptoas_path", lambda: ("/fake/ptoas", os.environ.copy()))
    monkeypatch.setattr(compiler, "_get_objcopy_path", lambda: ("/fake/objcopy", os.environ.copy()))
    monkeypatch.setattr(compiler, "_get_aicore_linker_path", lambda: ("/fake/ld.lld", os.environ.copy()))

    def fake_run(cmd, env=None, stdout=None, stderr=None, check=False, **kwargs):
        commands.append(cmd)
        if cmd[0] == "/fake/ptoas":
            with open(cmd[-1], "wb") as f:
                f.write(b"host-fat-object")
        elif cmd[0] == "/fake/objcopy":
            section_arg = cmd[2]
            extracted_path = section_arg.split("=", 1)[1]
            with open(extracted_path, "wb") as f:
                f.write(b"aicore-rel")
            with open(cmd[-1], "wb") as f:
                f.write(b"fat-copy")
        elif cmd[0] == "/fake/ld.lld":
            with open(cmd[-1], "wb") as f:
                f.write(b"raw-npubin")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)

    result = compiler.ptoas_vmi_to_npubin("module { pto.vmi.return }\n", metadata, options)

    assert result == b"raw-npubin"
    assert commands[0][:3] == ["/fake/ptoas", "--pto-backend=vpto", "--pto-arch=a5"]
    assert commands[1][0] == "/fake/objcopy"
    assert commands[1][1] == "--dump-section"
    assert commands[1][2].startswith("__aicore_rel_binary=")
    assert commands[2][:5] == ["/fake/ld.lld", "-m", "aicorelinux", "-Ttext", "0"]
    assert "--allow-multiple-definition" in commands[2]
