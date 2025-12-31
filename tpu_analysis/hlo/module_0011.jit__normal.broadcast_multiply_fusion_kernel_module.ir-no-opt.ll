; ModuleID = '__compute_module_broadcast_multiply_fusion_kernel_module'
source_filename = "__compute_module_broadcast_multiply_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

declare float @xla.log1p.f32(float)

; Function Attrs: uwtable
define ptr @broadcast_multiply_fusion(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !4
  %10 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 1
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  %16 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 2
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  call void @broadcast_multiply_fusion_wrapped(ptr %5, ptr %7, ptr %9, i64 %13, i64 %15, i64 %17)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @broadcast_multiply_fusion_wrapped(ptr noalias align 64 dereferenceable(16384) %0, ptr noalias align 64 dereferenceable(16384) %1, ptr noalias align 64 dereferenceable(16384) %2, i64 %3, i64 %4, i64 %5) #1 {
  %7 = icmp sle i64 %3, 4
  br i1 %7, label %8, label %81

8:                                                ; preds = %6
  %9 = icmp sge i64 %3, 0
  br i1 %9, label %10, label %80

10:                                               ; preds = %8
  %11 = mul nsw i64 %3, 704
  br label %12

12:                                               ; preds = %77, %10
  %13 = phi i64 [ %78, %77 ], [ 0, %10 ]
  %14 = icmp slt i64 %13, 11
  br i1 %14, label %15, label %79

15:                                               ; preds = %12
  %16 = mul nsw i64 %13, 64
  %17 = add nsw i64 %11, %16
  br label %18

18:                                               ; preds = %21, %15
  %19 = phi i64 [ %76, %21 ], [ 0, %15 ]
  %20 = icmp slt i64 %19, 64
  br i1 %20, label %21, label %77

21:                                               ; preds = %18
  %22 = add nsw i64 %17, %19
  %23 = getelementptr inbounds [4096 x i32], ptr %0, i32 0, i64 %22
  %24 = load i32, ptr %23, align 4, !invariant.load !3
  %25 = getelementptr inbounds [4096 x i32], ptr %1, i32 0, i64 %22
  %26 = load i32, ptr %25, align 4, !invariant.load !3
  %27 = xor i32 %24, %26
  %28 = lshr i32 %27, 9
  %29 = or i32 %28, 1065353216
  %30 = bitcast i32 %29 to float
  %31 = fadd float %30, -1.000000e+00
  %32 = fmul float %31, 2.000000e+00
  %33 = fadd float %32, 0xBFEFFFFFE0000000
  %34 = call float @llvm.maximum.f32(float %33, float 0xBFEFFFFFE0000000)
  %35 = fneg float %34
  %36 = fmul float %34, %35
  %37 = call float @xla.log1p.f32(float %36)
  %38 = fneg float %37
  %39 = fcmp olt float %38, 5.000000e+00
  %40 = select i1 %39, float 0x3E5E2CB100000000, float 0xBF2A3E1360000000
  %41 = select i1 %39, float 0x3E970966C0000000, float 0x3F1A76AD60000000
  %42 = call float @llvm.sqrt.f32(float %38)
  %43 = fadd float %38, -2.500000e+00
  %44 = fadd float %42, -3.000000e+00
  %45 = select i1 %39, float %43, float %44
  %46 = fmul float %40, %45
  %47 = fadd float %41, %46
  %48 = select i1 %39, float 0xBECD8E6AE0000000, float 0x3F561B8E40000000
  %49 = fmul float %47, %45
  %50 = fadd float %48, %49
  %51 = select i1 %39, float 0xBED26B5820000000, float 0xBF6E17BCE0000000
  %52 = fmul float %50, %45
  %53 = fadd float %51, %52
  %54 = select i1 %39, float 0x3F2CA65B60000000, float 0x3F77824F60000000
  %55 = fmul float %53, %45
  %56 = fadd float %54, %55
  %57 = select i1 %39, float 0xBF548A8100000000, float 0xBF7F38BAE0000000
  %58 = fmul float %56, %45
  %59 = fadd float %57, %58
  %60 = select i1 %39, float 0xBF711C9DE0000000, float 0x3F8354AFC0000000
  %61 = fmul float %59, %45
  %62 = fadd float %60, %61
  %63 = select i1 %39, float 0x3FCF91EC60000000, float 0x3FF006DB60000000
  %64 = fmul float %62, %45
  %65 = fadd float %63, %64
  %66 = select i1 %39, float 0x3FF805C5E0000000, float 0x4006A9EFC0000000
  %67 = fmul float %65, %45
  %68 = call float @llvm.fabs.f32(float %34)
  %69 = fadd float %66, %67
  %70 = fcmp oeq float %68, 1.000000e+00
  %71 = fmul float %34, 0x7FF0000000000000
  %72 = fmul float %69, %34
  %73 = select i1 %70, float %71, float %72
  %74 = fmul float %73, 0x3FF6A09E60000000
  %75 = getelementptr inbounds [4096 x float], ptr %2, i32 0, i64 %22
  store float %74, ptr %75, align 4
  %76 = add i64 %19, 1
  br label %18

77:                                               ; preds = %18
  %78 = add i64 %13, 1
  br label %12, !llvm.loop !5

79:                                               ; preds = %12, %84
  br label %80

80:                                               ; preds = %79, %8, %81
  br label %151

81:                                               ; preds = %6
  %82 = icmp eq i64 %3, 5
  br i1 %82, label %83, label %80

83:                                               ; preds = %81
  br label %84

84:                                               ; preds = %149, %83
  %85 = phi i64 [ %150, %149 ], [ 0, %83 ]
  %86 = icmp slt i64 %85, 9
  br i1 %86, label %87, label %79

87:                                               ; preds = %84
  %88 = mul nsw i64 %85, 64
  br label %89

89:                                               ; preds = %92, %87
  %90 = phi i64 [ %148, %92 ], [ 0, %87 ]
  %91 = icmp slt i64 %90, 64
  br i1 %91, label %92, label %149

92:                                               ; preds = %89
  %93 = add nsw i64 %88, %90
  %94 = add nsw i64 %93, 3520
  %95 = getelementptr inbounds [4096 x i32], ptr %0, i32 0, i64 %94
  %96 = load i32, ptr %95, align 4, !invariant.load !3
  %97 = getelementptr inbounds [4096 x i32], ptr %1, i32 0, i64 %94
  %98 = load i32, ptr %97, align 4, !invariant.load !3
  %99 = xor i32 %96, %98
  %100 = lshr i32 %99, 9
  %101 = or i32 %100, 1065353216
  %102 = bitcast i32 %101 to float
  %103 = fadd float %102, -1.000000e+00
  %104 = fmul float %103, 2.000000e+00
  %105 = fadd float %104, 0xBFEFFFFFE0000000
  %106 = call float @llvm.maximum.f32(float %105, float 0xBFEFFFFFE0000000)
  %107 = fneg float %106
  %108 = fmul float %106, %107
  %109 = call float @xla.log1p.f32(float %108)
  %110 = fneg float %109
  %111 = fcmp olt float %110, 5.000000e+00
  %112 = select i1 %111, float 0x3E5E2CB100000000, float 0xBF2A3E1360000000
  %113 = select i1 %111, float 0x3E970966C0000000, float 0x3F1A76AD60000000
  %114 = call float @llvm.sqrt.f32(float %110)
  %115 = fadd float %110, -2.500000e+00
  %116 = fadd float %114, -3.000000e+00
  %117 = select i1 %111, float %115, float %116
  %118 = fmul float %112, %117
  %119 = fadd float %113, %118
  %120 = select i1 %111, float 0xBECD8E6AE0000000, float 0x3F561B8E40000000
  %121 = fmul float %119, %117
  %122 = fadd float %120, %121
  %123 = select i1 %111, float 0xBED26B5820000000, float 0xBF6E17BCE0000000
  %124 = fmul float %122, %117
  %125 = fadd float %123, %124
  %126 = select i1 %111, float 0x3F2CA65B60000000, float 0x3F77824F60000000
  %127 = fmul float %125, %117
  %128 = fadd float %126, %127
  %129 = select i1 %111, float 0xBF548A8100000000, float 0xBF7F38BAE0000000
  %130 = fmul float %128, %117
  %131 = fadd float %129, %130
  %132 = select i1 %111, float 0xBF711C9DE0000000, float 0x3F8354AFC0000000
  %133 = fmul float %131, %117
  %134 = fadd float %132, %133
  %135 = select i1 %111, float 0x3FCF91EC60000000, float 0x3FF006DB60000000
  %136 = fmul float %134, %117
  %137 = fadd float %135, %136
  %138 = select i1 %111, float 0x3FF805C5E0000000, float 0x4006A9EFC0000000
  %139 = fmul float %137, %117
  %140 = call float @llvm.fabs.f32(float %106)
  %141 = fadd float %138, %139
  %142 = fcmp oeq float %140, 1.000000e+00
  %143 = fmul float %106, 0x7FF0000000000000
  %144 = fmul float %141, %106
  %145 = select i1 %142, float %143, float %144
  %146 = fmul float %145, 0x3FF6A09E60000000
  %147 = getelementptr inbounds [4096 x float], ptr %2, i32 0, i64 %94
  store float %146, ptr %147, align 4
  %148 = add i64 %90, 1
  br label %89

149:                                              ; preds = %89
  %150 = add i64 %85, 1
  br label %84, !llvm.loop !5

151:                                              ; preds = %80
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 4}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 16384}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
