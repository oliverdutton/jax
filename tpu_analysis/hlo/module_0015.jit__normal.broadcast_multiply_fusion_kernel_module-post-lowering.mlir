module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @xla.log1p.f32(f32) -> f32 attributes {sym_visibility = "private"}
  llvm.func @broadcast_multiply_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 8192> : !llvm.ptr -> !llvm.ptr
    %9 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %12 = llvm.load %11 invariant : !llvm.ptr -> i64
    %13 = llvm.getelementptr inbounds %10[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %14 = llvm.load %13 invariant : !llvm.ptr -> i64
    %15 = llvm.getelementptr inbounds %10[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"kernel_dim3", (i64, i64, i64)>
    %16 = llvm.load %15 invariant : !llvm.ptr -> i64
    llvm.call @broadcast_multiply_fusion_wrapped(%4, %6, %8, %12, %14, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @broadcast_multiply_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 8192 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(1408 : index) : i64
    %1 = llvm.mlir.constant(704 : index) : i64
    %2 = llvm.mlir.constant(2 : index) : i64
    %3 = llvm.mlir.constant(20 : index) : i64
    %4 = llvm.mlir.constant(32 : index) : i64
    %5 = llvm.mlir.constant(22 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(1.00950558E-4 : f32) : f32
    %9 = llvm.mlir.constant(3.43273939E-7 : f32) : f32
    %10 = llvm.mlir.constant(-2.00214257E-4 : f32) : f32
    %11 = llvm.mlir.constant(2.81022636E-8 : f32) : f32
    %12 = llvm.mlir.constant(9 : i32) : i32
    %13 = llvm.mlir.constant(1065353216 : i32) : i32
    %14 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %15 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %16 = llvm.mlir.constant(-0.99999994 : f32) : f32
    %17 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %18 = llvm.mlir.constant(-2.500000e+00 : f32) : f32
    %19 = llvm.mlir.constant(-3.000000e+00 : f32) : f32
    %20 = llvm.mlir.constant(-3.5233877E-6 : f32) : f32
    %21 = llvm.mlir.constant(0.00134934322 : f32) : f32
    %22 = llvm.mlir.constant(-4.39150654E-6 : f32) : f32
    %23 = llvm.mlir.constant(-0.00367342844 : f32) : f32
    %24 = llvm.mlir.constant(2.1858087E-4 : f32) : f32
    %25 = llvm.mlir.constant(0.00573950773 : f32) : f32
    %26 = llvm.mlir.constant(-0.00125372503 : f32) : f32
    %27 = llvm.mlir.constant(-0.0076224613 : f32) : f32
    %28 = llvm.mlir.constant(-0.00417768164 : f32) : f32
    %29 = llvm.mlir.constant(0.00943887047 : f32) : f32
    %30 = llvm.mlir.constant(0.246640727 : f32) : f32
    %31 = llvm.mlir.constant(1.00167406 : f32) : f32
    %32 = llvm.mlir.constant(1.50140941 : f32) : f32
    %33 = llvm.mlir.constant(2.83297682 : f32) : f32
    %34 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %35 = llvm.mlir.constant(0x7F800000 : f32) : f32
    %36 = llvm.mlir.constant(1.41421354 : f32) : f32
    %37 = llvm.icmp "sle" %arg3, %7 : i64
    llvm.cond_br %37, ^bb1, ^bb10
  ^bb1:  // pred: ^bb0
    %38 = llvm.icmp "sge" %arg3, %6 : i64
    llvm.cond_br %38, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %39 = llvm.mul %arg3, %1 overflow<nsw> : i64
    llvm.br ^bb3(%6 : i64)
  ^bb3(%40: i64):  // 2 preds: ^bb2, ^bb7
    %41 = llvm.icmp "slt" %40, %5 : i64
    llvm.cond_br %41, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %42 = llvm.mul %40, %4 overflow<nsw> : i64
    %43 = llvm.add %39, %42 overflow<nsw> : i64
    llvm.br ^bb5(%6 : i64)
  ^bb5(%44: i64):  // 2 preds: ^bb4, ^bb6
    %45 = llvm.icmp "slt" %44, %4 : i64
    llvm.cond_br %45, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %46 = llvm.add %43, %44 overflow<nsw> : i64
    %47 = llvm.getelementptr inbounds %arg0[0, %46] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %48 = llvm.load %47 invariant : !llvm.ptr -> i32
    %49 = llvm.getelementptr inbounds %arg1[0, %46] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %50 = llvm.load %49 invariant : !llvm.ptr -> i32
    %51 = llvm.xor %48, %50 : i32
    %52 = llvm.lshr %51, %12 : i32
    %53 = llvm.or %52, %13 : i32
    %54 = llvm.bitcast %53 : i32 to f32
    %55 = llvm.fadd %54, %14 : f32
    %56 = llvm.fmul %55, %15 : f32
    %57 = llvm.fadd %56, %16 : f32
    %58 = llvm.intr.maximum(%57, %16) : (f32, f32) -> f32
    %59 = llvm.fneg %58 : f32
    %60 = llvm.fmul %58, %59 : f32
    %61 = llvm.call @xla.log1p.f32(%60) : (f32) -> f32
    %62 = llvm.fneg %61 : f32
    %63 = llvm.fcmp "olt" %62, %17 : f32
    %64 = llvm.select %63, %11, %10 : i1, f32
    %65 = llvm.select %63, %9, %8 : i1, f32
    %66 = llvm.intr.sqrt(%62) : (f32) -> f32
    %67 = llvm.fadd %62, %18 : f32
    %68 = llvm.fadd %66, %19 : f32
    %69 = llvm.select %63, %67, %68 : i1, f32
    %70 = llvm.fmul %64, %69 : f32
    %71 = llvm.fadd %65, %70 : f32
    %72 = llvm.select %63, %20, %21 : i1, f32
    %73 = llvm.fmul %71, %69 : f32
    %74 = llvm.fadd %72, %73 : f32
    %75 = llvm.select %63, %22, %23 : i1, f32
    %76 = llvm.fmul %74, %69 : f32
    %77 = llvm.fadd %75, %76 : f32
    %78 = llvm.select %63, %24, %25 : i1, f32
    %79 = llvm.fmul %77, %69 : f32
    %80 = llvm.fadd %78, %79 : f32
    %81 = llvm.select %63, %26, %27 : i1, f32
    %82 = llvm.fmul %80, %69 : f32
    %83 = llvm.fadd %81, %82 : f32
    %84 = llvm.select %63, %28, %29 : i1, f32
    %85 = llvm.fmul %83, %69 : f32
    %86 = llvm.fadd %84, %85 : f32
    %87 = llvm.select %63, %30, %31 : i1, f32
    %88 = llvm.fmul %86, %69 : f32
    %89 = llvm.fadd %87, %88 : f32
    %90 = llvm.select %63, %32, %33 : i1, f32
    %91 = llvm.fmul %89, %69 : f32
    %92 = llvm.intr.fabs(%58) : (f32) -> f32
    %93 = llvm.fadd %90, %91 : f32
    %94 = llvm.fcmp "oeq" %92, %34 : f32
    %95 = llvm.fmul %58, %35 : f32
    %96 = llvm.fmul %93, %58 : f32
    %97 = llvm.select %94, %95, %96 : i1, f32
    %98 = llvm.fmul %97, %36 : f32
    %99 = llvm.getelementptr inbounds %arg2[0, %46] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x f32>
    llvm.store %98, %99 : f32, !llvm.ptr
    %100 = llvm.add %44, %7 : i64
    llvm.br ^bb5(%100 : i64)
  ^bb7:  // pred: ^bb5
    %101 = llvm.add %40, %7 : i64
    llvm.br ^bb3(%101 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // 2 preds: ^bb3, ^bb12
    llvm.br ^bb9
  ^bb9:  // 3 preds: ^bb1, ^bb8, ^bb10
    llvm.br ^bb17
  ^bb10:  // pred: ^bb0
    %102 = llvm.icmp "eq" %arg3, %2 : i64
    llvm.cond_br %102, ^bb11, ^bb9
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%6 : i64)
  ^bb12(%103: i64):  // 2 preds: ^bb11, ^bb16
    %104 = llvm.icmp "slt" %103, %3 : i64
    llvm.cond_br %104, ^bb13, ^bb8
  ^bb13:  // pred: ^bb12
    %105 = llvm.mul %103, %4 overflow<nsw> : i64
    llvm.br ^bb14(%6 : i64)
  ^bb14(%106: i64):  // 2 preds: ^bb13, ^bb15
    %107 = llvm.icmp "slt" %106, %4 : i64
    llvm.cond_br %107, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %108 = llvm.add %105, %106 overflow<nsw> : i64
    %109 = llvm.add %108, %0 overflow<nsw> : i64
    %110 = llvm.getelementptr inbounds %arg0[0, %109] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %111 = llvm.load %110 invariant : !llvm.ptr -> i32
    %112 = llvm.getelementptr inbounds %arg1[0, %109] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x i32>
    %113 = llvm.load %112 invariant : !llvm.ptr -> i32
    %114 = llvm.xor %111, %113 : i32
    %115 = llvm.lshr %114, %12 : i32
    %116 = llvm.or %115, %13 : i32
    %117 = llvm.bitcast %116 : i32 to f32
    %118 = llvm.fadd %117, %14 : f32
    %119 = llvm.fmul %118, %15 : f32
    %120 = llvm.fadd %119, %16 : f32
    %121 = llvm.intr.maximum(%120, %16) : (f32, f32) -> f32
    %122 = llvm.fneg %121 : f32
    %123 = llvm.fmul %121, %122 : f32
    %124 = llvm.call @xla.log1p.f32(%123) : (f32) -> f32
    %125 = llvm.fneg %124 : f32
    %126 = llvm.fcmp "olt" %125, %17 : f32
    %127 = llvm.select %126, %11, %10 : i1, f32
    %128 = llvm.select %126, %9, %8 : i1, f32
    %129 = llvm.intr.sqrt(%125) : (f32) -> f32
    %130 = llvm.fadd %125, %18 : f32
    %131 = llvm.fadd %129, %19 : f32
    %132 = llvm.select %126, %130, %131 : i1, f32
    %133 = llvm.fmul %127, %132 : f32
    %134 = llvm.fadd %128, %133 : f32
    %135 = llvm.select %126, %20, %21 : i1, f32
    %136 = llvm.fmul %134, %132 : f32
    %137 = llvm.fadd %135, %136 : f32
    %138 = llvm.select %126, %22, %23 : i1, f32
    %139 = llvm.fmul %137, %132 : f32
    %140 = llvm.fadd %138, %139 : f32
    %141 = llvm.select %126, %24, %25 : i1, f32
    %142 = llvm.fmul %140, %132 : f32
    %143 = llvm.fadd %141, %142 : f32
    %144 = llvm.select %126, %26, %27 : i1, f32
    %145 = llvm.fmul %143, %132 : f32
    %146 = llvm.fadd %144, %145 : f32
    %147 = llvm.select %126, %28, %29 : i1, f32
    %148 = llvm.fmul %146, %132 : f32
    %149 = llvm.fadd %147, %148 : f32
    %150 = llvm.select %126, %30, %31 : i1, f32
    %151 = llvm.fmul %149, %132 : f32
    %152 = llvm.fadd %150, %151 : f32
    %153 = llvm.select %126, %32, %33 : i1, f32
    %154 = llvm.fmul %152, %132 : f32
    %155 = llvm.intr.fabs(%121) : (f32) -> f32
    %156 = llvm.fadd %153, %154 : f32
    %157 = llvm.fcmp "oeq" %155, %34 : f32
    %158 = llvm.fmul %121, %35 : f32
    %159 = llvm.fmul %156, %121 : f32
    %160 = llvm.select %157, %158, %159 : i1, f32
    %161 = llvm.fmul %160, %36 : f32
    %162 = llvm.getelementptr inbounds %arg2[0, %109] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2048 x f32>
    llvm.store %161, %162 : f32, !llvm.ptr
    %163 = llvm.add %106, %7 : i64
    llvm.br ^bb14(%163 : i64)
  ^bb16:  // pred: ^bb14
    %164 = llvm.add %103, %7 : i64
    llvm.br ^bb12(%164 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb17:  // pred: ^bb9
    llvm.return
  }
}