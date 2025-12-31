module @broadcast_multiply_fusion_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>, xla.cpu_memory_region_name = "xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"} {
  llvm.func @xla.log1p.f32(f32) -> f32 attributes {sym_visibility = "private"}
  llvm.func @broadcast_multiply_fusion(%arg0: !llvm.ptr) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["prefer-vector-width", "256"]], uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr inbounds %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelCallFrame", (ptr, ptr, i64, ptr)>
    %2 = llvm.load %1 invariant : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %4 = llvm.load %3 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %2[1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %6 = llvm.load %5 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %2[2, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"XLA_CPU_KernelArg", (ptr, i64)>
    %8 = llvm.load %7 invariant dereferenceable<bytes = 16384> : !llvm.ptr -> !llvm.ptr
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
  llvm.func internal @broadcast_multiply_fusion_wrapped(%arg0: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias, xla.invariant}, %arg1: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias, xla.invariant}, %arg2: !llvm.ptr {llvm.align = 64 : index, llvm.dereferenceable = 16384 : index, llvm.noalias}, %arg3: i64, %arg4: i64, %arg5: i64) attributes {always_inline, sym_visibility = "private", xla.backend_kind = #xla.backend_kind<cpu>, xla.cpu.is_wrapped, xla.entry} {
    %0 = llvm.mlir.constant(3520 : index) : i64
    %1 = llvm.mlir.constant(704 : index) : i64
    %2 = llvm.mlir.constant(5 : index) : i64
    %3 = llvm.mlir.constant(9 : index) : i64
    %4 = llvm.mlir.constant(64 : index) : i64
    %5 = llvm.mlir.constant(11 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(4 : index) : i64
    %9 = llvm.mlir.constant(1.00950558E-4 : f32) : f32
    %10 = llvm.mlir.constant(3.43273939E-7 : f32) : f32
    %11 = llvm.mlir.constant(-2.00214257E-4 : f32) : f32
    %12 = llvm.mlir.constant(2.81022636E-8 : f32) : f32
    %13 = llvm.mlir.constant(9 : i32) : i32
    %14 = llvm.mlir.constant(1065353216 : i32) : i32
    %15 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %16 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %17 = llvm.mlir.constant(-0.99999994 : f32) : f32
    %18 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %19 = llvm.mlir.constant(-2.500000e+00 : f32) : f32
    %20 = llvm.mlir.constant(-3.000000e+00 : f32) : f32
    %21 = llvm.mlir.constant(-3.5233877E-6 : f32) : f32
    %22 = llvm.mlir.constant(0.00134934322 : f32) : f32
    %23 = llvm.mlir.constant(-4.39150654E-6 : f32) : f32
    %24 = llvm.mlir.constant(-0.00367342844 : f32) : f32
    %25 = llvm.mlir.constant(2.1858087E-4 : f32) : f32
    %26 = llvm.mlir.constant(0.00573950773 : f32) : f32
    %27 = llvm.mlir.constant(-0.00125372503 : f32) : f32
    %28 = llvm.mlir.constant(-0.0076224613 : f32) : f32
    %29 = llvm.mlir.constant(-0.00417768164 : f32) : f32
    %30 = llvm.mlir.constant(0.00943887047 : f32) : f32
    %31 = llvm.mlir.constant(0.246640727 : f32) : f32
    %32 = llvm.mlir.constant(1.00167406 : f32) : f32
    %33 = llvm.mlir.constant(1.50140941 : f32) : f32
    %34 = llvm.mlir.constant(2.83297682 : f32) : f32
    %35 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %36 = llvm.mlir.constant(0x7F800000 : f32) : f32
    %37 = llvm.mlir.constant(1.41421354 : f32) : f32
    %38 = llvm.icmp "sle" %arg3, %8 : i64
    llvm.cond_br %38, ^bb1, ^bb10
  ^bb1:  // pred: ^bb0
    %39 = llvm.icmp "sge" %arg3, %6 : i64
    llvm.cond_br %39, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %40 = llvm.mul %arg3, %1 overflow<nsw> : i64
    llvm.br ^bb3(%6 : i64)
  ^bb3(%41: i64):  // 2 preds: ^bb2, ^bb7
    %42 = llvm.icmp "slt" %41, %5 : i64
    llvm.cond_br %42, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %43 = llvm.mul %41, %4 overflow<nsw> : i64
    %44 = llvm.add %40, %43 overflow<nsw> : i64
    llvm.br ^bb5(%6 : i64)
  ^bb5(%45: i64):  // 2 preds: ^bb4, ^bb6
    %46 = llvm.icmp "slt" %45, %4 : i64
    llvm.cond_br %46, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %47 = llvm.add %44, %45 overflow<nsw> : i64
    %48 = llvm.getelementptr inbounds %arg0[0, %47] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %49 = llvm.load %48 invariant : !llvm.ptr -> i32
    %50 = llvm.getelementptr inbounds %arg1[0, %47] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %51 = llvm.load %50 invariant : !llvm.ptr -> i32
    %52 = llvm.xor %49, %51 : i32
    %53 = llvm.lshr %52, %13 : i32
    %54 = llvm.or %53, %14 : i32
    %55 = llvm.bitcast %54 : i32 to f32
    %56 = llvm.fadd %55, %15 : f32
    %57 = llvm.fmul %56, %16 : f32
    %58 = llvm.fadd %57, %17 : f32
    %59 = llvm.intr.maximum(%58, %17) : (f32, f32) -> f32
    %60 = llvm.fneg %59 : f32
    %61 = llvm.fmul %59, %60 : f32
    %62 = llvm.call @xla.log1p.f32(%61) : (f32) -> f32
    %63 = llvm.fneg %62 : f32
    %64 = llvm.fcmp "olt" %63, %18 : f32
    %65 = llvm.select %64, %12, %11 : i1, f32
    %66 = llvm.select %64, %10, %9 : i1, f32
    %67 = llvm.intr.sqrt(%63) : (f32) -> f32
    %68 = llvm.fadd %63, %19 : f32
    %69 = llvm.fadd %67, %20 : f32
    %70 = llvm.select %64, %68, %69 : i1, f32
    %71 = llvm.fmul %65, %70 : f32
    %72 = llvm.fadd %66, %71 : f32
    %73 = llvm.select %64, %21, %22 : i1, f32
    %74 = llvm.fmul %72, %70 : f32
    %75 = llvm.fadd %73, %74 : f32
    %76 = llvm.select %64, %23, %24 : i1, f32
    %77 = llvm.fmul %75, %70 : f32
    %78 = llvm.fadd %76, %77 : f32
    %79 = llvm.select %64, %25, %26 : i1, f32
    %80 = llvm.fmul %78, %70 : f32
    %81 = llvm.fadd %79, %80 : f32
    %82 = llvm.select %64, %27, %28 : i1, f32
    %83 = llvm.fmul %81, %70 : f32
    %84 = llvm.fadd %82, %83 : f32
    %85 = llvm.select %64, %29, %30 : i1, f32
    %86 = llvm.fmul %84, %70 : f32
    %87 = llvm.fadd %85, %86 : f32
    %88 = llvm.select %64, %31, %32 : i1, f32
    %89 = llvm.fmul %87, %70 : f32
    %90 = llvm.fadd %88, %89 : f32
    %91 = llvm.select %64, %33, %34 : i1, f32
    %92 = llvm.fmul %90, %70 : f32
    %93 = llvm.intr.fabs(%59) : (f32) -> f32
    %94 = llvm.fadd %91, %92 : f32
    %95 = llvm.fcmp "oeq" %93, %35 : f32
    %96 = llvm.fmul %59, %36 : f32
    %97 = llvm.fmul %94, %59 : f32
    %98 = llvm.select %95, %96, %97 : i1, f32
    %99 = llvm.fmul %98, %37 : f32
    %100 = llvm.getelementptr inbounds %arg2[0, %47] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x f32>
    llvm.store %99, %100 : f32, !llvm.ptr
    %101 = llvm.add %45, %7 : i64
    llvm.br ^bb5(%101 : i64)
  ^bb7:  // pred: ^bb5
    %102 = llvm.add %41, %7 : i64
    llvm.br ^bb3(%102 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb8:  // 2 preds: ^bb3, ^bb12
    llvm.br ^bb9
  ^bb9:  // 3 preds: ^bb1, ^bb8, ^bb10
    llvm.br ^bb17
  ^bb10:  // pred: ^bb0
    %103 = llvm.icmp "eq" %arg3, %2 : i64
    llvm.cond_br %103, ^bb11, ^bb9
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%6 : i64)
  ^bb12(%104: i64):  // 2 preds: ^bb11, ^bb16
    %105 = llvm.icmp "slt" %104, %3 : i64
    llvm.cond_br %105, ^bb13, ^bb8
  ^bb13:  // pred: ^bb12
    %106 = llvm.mul %104, %4 overflow<nsw> : i64
    llvm.br ^bb14(%6 : i64)
  ^bb14(%107: i64):  // 2 preds: ^bb13, ^bb15
    %108 = llvm.icmp "slt" %107, %4 : i64
    llvm.cond_br %108, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %109 = llvm.add %106, %107 overflow<nsw> : i64
    %110 = llvm.add %109, %0 overflow<nsw> : i64
    %111 = llvm.getelementptr inbounds %arg0[0, %110] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %112 = llvm.load %111 invariant : !llvm.ptr -> i32
    %113 = llvm.getelementptr inbounds %arg1[0, %110] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x i32>
    %114 = llvm.load %113 invariant : !llvm.ptr -> i32
    %115 = llvm.xor %112, %114 : i32
    %116 = llvm.lshr %115, %13 : i32
    %117 = llvm.or %116, %14 : i32
    %118 = llvm.bitcast %117 : i32 to f32
    %119 = llvm.fadd %118, %15 : f32
    %120 = llvm.fmul %119, %16 : f32
    %121 = llvm.fadd %120, %17 : f32
    %122 = llvm.intr.maximum(%121, %17) : (f32, f32) -> f32
    %123 = llvm.fneg %122 : f32
    %124 = llvm.fmul %122, %123 : f32
    %125 = llvm.call @xla.log1p.f32(%124) : (f32) -> f32
    %126 = llvm.fneg %125 : f32
    %127 = llvm.fcmp "olt" %126, %18 : f32
    %128 = llvm.select %127, %12, %11 : i1, f32
    %129 = llvm.select %127, %10, %9 : i1, f32
    %130 = llvm.intr.sqrt(%126) : (f32) -> f32
    %131 = llvm.fadd %126, %19 : f32
    %132 = llvm.fadd %130, %20 : f32
    %133 = llvm.select %127, %131, %132 : i1, f32
    %134 = llvm.fmul %128, %133 : f32
    %135 = llvm.fadd %129, %134 : f32
    %136 = llvm.select %127, %21, %22 : i1, f32
    %137 = llvm.fmul %135, %133 : f32
    %138 = llvm.fadd %136, %137 : f32
    %139 = llvm.select %127, %23, %24 : i1, f32
    %140 = llvm.fmul %138, %133 : f32
    %141 = llvm.fadd %139, %140 : f32
    %142 = llvm.select %127, %25, %26 : i1, f32
    %143 = llvm.fmul %141, %133 : f32
    %144 = llvm.fadd %142, %143 : f32
    %145 = llvm.select %127, %27, %28 : i1, f32
    %146 = llvm.fmul %144, %133 : f32
    %147 = llvm.fadd %145, %146 : f32
    %148 = llvm.select %127, %29, %30 : i1, f32
    %149 = llvm.fmul %147, %133 : f32
    %150 = llvm.fadd %148, %149 : f32
    %151 = llvm.select %127, %31, %32 : i1, f32
    %152 = llvm.fmul %150, %133 : f32
    %153 = llvm.fadd %151, %152 : f32
    %154 = llvm.select %127, %33, %34 : i1, f32
    %155 = llvm.fmul %153, %133 : f32
    %156 = llvm.intr.fabs(%122) : (f32) -> f32
    %157 = llvm.fadd %154, %155 : f32
    %158 = llvm.fcmp "oeq" %156, %35 : f32
    %159 = llvm.fmul %122, %36 : f32
    %160 = llvm.fmul %157, %122 : f32
    %161 = llvm.select %158, %159, %160 : i1, f32
    %162 = llvm.fmul %161, %37 : f32
    %163 = llvm.getelementptr inbounds %arg2[0, %110] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4096 x f32>
    llvm.store %162, %163 : f32, !llvm.ptr
    %164 = llvm.add %107, %7 : i64
    llvm.br ^bb14(%164 : i64)
  ^bb16:  // pred: ^bb14
    %165 = llvm.add %104, %7 : i64
    llvm.br ^bb12(%165 : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  ^bb17:  // pred: ^bb9
    llvm.return
  }
}