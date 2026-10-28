// Minimal mainline compiler fixture: no templates, SET2D, or PIPE_S.
module attributes {mix_mode = "mix", parallel_mode = "simd"} {
  func.func @vector_add_kernel() attributes {
      hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>,
      hacc.infer_workspace_shape_function = #hacc.infer_workspace_shape_function<@vector_add_kernel_infer_workspace_shape_function>} {
    return
  }
  func.func @vector_add_kernel_mix_aic() attributes {
      hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    return
  }
  func.func @vector_add_kernel_infer_workspace_shape_function() -> index attributes {
      hacc.function_kind = #hacc.function_kind<HOST>,
      hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %n = func.call @host_workspace_helper() : () -> index
    return %n : index
  }
  func.func private @host_workspace_helper() -> index attributes {
      hacc.function_kind = #hacc.function_kind<HOST>} {
    %n = arith.constant 16 : index
    return %n : index
  }
}
