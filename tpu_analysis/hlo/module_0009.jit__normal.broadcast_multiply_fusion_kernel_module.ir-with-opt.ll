; ModuleID = '__compute_module_broadcast_multiply_fusion_kernel_module'
source_filename = "__compute_module_broadcast_multiply_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define noalias noundef ptr @broadcast_multiply_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !4
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  br label %vector.ph

vector.ph:                                        ; preds = %1, %middle.block
  %9 = phi i64 [ 0, %1 ], [ %110, %middle.block ]
  %10 = shl nuw nsw i64 %9, 6
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %11 = add nuw nsw i64 %index, %10
  %12 = getelementptr inbounds nuw i32, ptr %4, i64 %11
  %wide.load = load <8 x i32>, ptr %12, align 4, !invariant.load !3, !alias.scope !5, !noalias !12
  %13 = getelementptr inbounds nuw i32, ptr %6, i64 %11
  %wide.load3 = load <8 x i32>, ptr %13, align 4, !invariant.load !3, !alias.scope !8, !noalias !13
  %14 = xor <8 x i32> %wide.load3, %wide.load
  %15 = lshr <8 x i32> %14, splat (i32 9)
  %16 = or disjoint <8 x i32> %15, splat (i32 1065353216)
  %17 = bitcast <8 x i32> %16 to <8 x float>
  %18 = fadd <8 x float> %17, splat (float -1.000000e+00)
  %19 = fmul <8 x float> %18, splat (float 2.000000e+00)
  %20 = fadd <8 x float> %19, splat (float 0xBFEFFFFFE0000000)
  %21 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %20, <8 x float> splat (float 0xBFEFFFFFE0000000))
  %22 = fneg <8 x float> %21
  %23 = fmul <8 x float> %21, %22
  %24 = fadd <8 x float> %23, splat (float 1.000000e+00)
  %log_f32.i.i = fcmp ule <8 x float> %24, zeroinitializer
  %log_f323.i.i = fcmp une <8 x float> %24, zeroinitializer
  %log_f326.i.i = fcmp une <8 x float> %24, splat (float 0x7FF0000000000000)
  %.inv = fcmp ogt <8 x float> %24, splat (float 0x3810000000000000)
  %25 = select <8 x i1> %.inv, <8 x float> %24, <8 x float> splat (float 0x3810000000000000)
  %26 = bitcast <8 x float> %25 to <8 x i32>
  %27 = lshr <8 x i32> %26, splat (i32 23)
  %log_f3210.i.i = and <8 x i32> %26, splat (i32 8388607)
  %log_f3212.i.i = or disjoint <8 x i32> %log_f3210.i.i, splat (i32 1056964608)
  %log_f3213.i.i = bitcast <8 x i32> %log_f3212.i.i to <8 x float>
  %28 = add nsw <8 x i32> %27, splat (i32 -127)
  %29 = sitofp <8 x i32> %28 to <8 x float>
  %log_f3214.i.i = fadd <8 x float> %29, splat (float 1.000000e+00)
  %log_f3215.i.i = fcmp olt <8 x float> %log_f3213.i.i, splat (float 0x3FE6A09E60000000)
  %30 = select <8 x i1> %log_f3215.i.i, <8 x float> %log_f3213.i.i, <8 x float> zeroinitializer
  %31 = fadd <8 x float> %log_f3213.i.i, splat (float -1.000000e+00)
  %32 = select <8 x i1> %log_f3215.i.i, <8 x float> splat (float 1.000000e+00), <8 x float> zeroinitializer
  %33 = fsub <8 x float> %log_f3214.i.i, %32
  %log_f3223.i.i = fadd <8 x float> %31, %30
  %log_f3224.i.i = fmul <8 x float> %log_f3223.i.i, %log_f3223.i.i
  %log_f3225.i.i = fmul <8 x float> %log_f3224.i.i, %log_f3223.i.i
  %log_f3226.i.i = fmul <8 x float> %log_f3223.i.i, splat (float 0x3FB2043760000000)
  %log_f3227.i.i = fadd <8 x float> %log_f3226.i.i, splat (float 0xBFBD7A3700000000)
  %log_f3228.i.i = fmul <8 x float> %log_f3223.i.i, splat (float 0xBFBFCBA9E0000000)
  %log_f3229.i.i = fadd <8 x float> %log_f3228.i.i, splat (float 0x3FC23D37E0000000)
  %log_f3230.i.i = fmul <8 x float> %log_f3223.i.i, splat (float 0x3FC999D580000000)
  %log_f3231.i.i = fadd <8 x float> %log_f3230.i.i, splat (float 0xBFCFFFFF80000000)
  %log_f3232.i.i = fmul <8 x float> %log_f3227.i.i, %log_f3223.i.i
  %log_f3233.i.i = fadd <8 x float> %log_f3232.i.i, splat (float 0x3FBDE4A340000000)
  %log_f3234.i.i = fmul <8 x float> %log_f3229.i.i, %log_f3223.i.i
  %log_f3235.i.i = fadd <8 x float> %log_f3234.i.i, splat (float 0xBFC555CA00000000)
  %log_f3236.i.i = fmul <8 x float> %log_f3231.i.i, %log_f3223.i.i
  %log_f3237.i.i = fadd <8 x float> %log_f3236.i.i, splat (float 0x3FD5555540000000)
  %log_f3238.i.i = fmul <8 x float> %log_f3233.i.i, %log_f3225.i.i
  %log_f3239.i.i = fadd <8 x float> %log_f3235.i.i, %log_f3238.i.i
  %log_f3240.i.i = fmul <8 x float> %log_f3239.i.i, %log_f3225.i.i
  %log_f3241.i.i = fadd <8 x float> %log_f3237.i.i, %log_f3240.i.i
  %log_f3242.i.i = fmul <8 x float> %log_f3241.i.i, %log_f3225.i.i
  %log_f3243.i.i = fmul <8 x float> %33, splat (float 0xBF2BD01060000000)
  %log_f3244.i.i = fmul <8 x float> %log_f3224.i.i, splat (float 5.000000e-01)
  %log_f3245.i.i = fadd <8 x float> %log_f3242.i.i, %log_f3243.i.i
  %34 = fsub <8 x float> %log_f3223.i.i, %log_f3244.i.i
  %log_f3246.i.i = fmul <8 x float> %33, splat (float 0x3FE6300000000000)
  %log_f3247.i.i = fadd <8 x float> %34, %log_f3245.i.i
  %log_f3248.i.i = fadd <8 x float> %log_f3247.i.i, %log_f3246.i.i
  %log_f3252.i.i = select <8 x i1> %log_f326.i.i, <8 x i32> zeroinitializer, <8 x i32> splat (i32 2139095040)
  %log_f3255.i.i = select <8 x i1> %log_f323.i.i, <8 x i32> %log_f3252.i.i, <8 x i32> splat (i32 -8388608)
  %log_f3257.i.i = bitcast <8 x float> %log_f3248.i.i to <8 x i32>
  %log_f3259.i.i = select <8 x i1> %log_f32.i.i, <8 x i32> splat (i32 -1), <8 x i32> %log_f3257.i.i
  %log_f3263.i.i5.not = and <8 x i1> %log_f323.i.i, %log_f326.i.i
  %log_f3269.i.i = select <8 x i1> %log_f3263.i.i5.not, <8 x i32> %log_f3259.i.i, <8 x i32> zeroinitializer
  %log_f3272.i.i = or <8 x i32> %log_f3255.i.i, %log_f3269.i.i
  %log_f3273.i.i = bitcast <8 x i32> %log_f3272.i.i to <8 x float>
  %35 = fmul <8 x float> %23, %23
  %36 = fmul <8 x float> %23, zeroinitializer
  %37 = fadd <8 x float> %36, splat (float 1.000000e+00)
  %38 = fmul <8 x float> %37, %23
  %39 = fadd <8 x float> %38, splat (float 0x402E2035A0000000)
  %40 = fmul <8 x float> %39, %23
  %41 = fadd <8 x float> %40, splat (float 0x4054C30B60000000)
  %42 = fmul <8 x float> %41, %23
  %43 = fadd <8 x float> %42, splat (float 0x406BB865A0000000)
  %44 = fmul <8 x float> %43, %23
  %45 = fadd <8 x float> %44, splat (float 0x4073519460000000)
  %46 = fmul <8 x float> %45, %23
  %47 = fadd <8 x float> %46, splat (float 0x406B0DB140000000)
  %48 = fmul <8 x float> %47, %23
  %49 = fadd <8 x float> %48, splat (float 0x404E0F3040000000)
  %50 = fadd <8 x float> %36, splat (float 0x3F07BC0960000000)
  %51 = fmul <8 x float> %50, %23
  %52 = fadd <8 x float> %51, splat (float 0x3FDFE818A0000000)
  %53 = fmul <8 x float> %52, %23
  %54 = fadd <8 x float> %53, splat (float 0x401A509F40000000)
  %55 = fmul <8 x float> %54, %23
  %56 = fadd <8 x float> %55, splat (float 0x403DE97380000000)
  %57 = fmul <8 x float> %56, %23
  %58 = fadd <8 x float> %57, splat (float 0x404E798EC0000000)
  %59 = fmul <8 x float> %58, %23
  %60 = fadd <8 x float> %59, splat (float 0x404C8E75A0000000)
  %61 = fmul <8 x float> %60, %23
  %62 = fadd <8 x float> %61, splat (float 0x40340A2020000000)
  %63 = fdiv <8 x float> %62, %49
  %64 = fmul <8 x float> %23, %35
  %65 = fmul <8 x float> %64, %63
  %66 = fmul <8 x float> %35, splat (float -5.000000e-01)
  %67 = fadd <8 x float> %66, %65
  %68 = fadd <8 x float> %23, %67
  %69 = call <8 x float> @llvm.fabs.v8f32(<8 x float> %23)
  %70 = fcmp olt <8 x float> %69, splat (float 0x3FDA8279A0000000)
  %71 = select <8 x i1> %70, <8 x float> %68, <8 x float> %log_f3273.i.i
  %72 = fneg <8 x float> %71
  %73 = fcmp ogt <8 x float> %71, splat (float -5.000000e+00)
  %74 = select <8 x i1> %73, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %75 = select <8 x i1> %73, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %76 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %72)
  %77 = fsub <8 x float> splat (float -2.500000e+00), %71
  %78 = fadd <8 x float> %76, splat (float -3.000000e+00)
  %79 = select <8 x i1> %73, <8 x float> %77, <8 x float> %78
  %80 = fmul <8 x float> %74, %79
  %81 = fadd <8 x float> %75, %80
  %82 = select <8 x i1> %73, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %83 = fmul <8 x float> %79, %81
  %84 = fadd <8 x float> %82, %83
  %85 = select <8 x i1> %73, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %86 = fmul <8 x float> %79, %84
  %87 = fadd <8 x float> %85, %86
  %88 = select <8 x i1> %73, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %89 = fmul <8 x float> %79, %87
  %90 = fadd <8 x float> %88, %89
  %91 = select <8 x i1> %73, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %92 = fmul <8 x float> %79, %90
  %93 = fadd <8 x float> %91, %92
  %94 = select <8 x i1> %73, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %95 = fmul <8 x float> %79, %93
  %96 = fadd <8 x float> %94, %95
  %97 = select <8 x i1> %73, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %98 = fmul <8 x float> %79, %96
  %99 = fadd <8 x float> %97, %98
  %100 = select <8 x i1> %73, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %101 = fmul <8 x float> %79, %99
  %102 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %21)
  %103 = fadd <8 x float> %100, %101
  %104 = fcmp oeq <8 x float> %102, splat (float 1.000000e+00)
  %105 = select <8 x i1> %104, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %103
  %106 = fmul <8 x float> %21, %105
  %107 = fmul <8 x float> %106, splat (float 0x3FF6A09E60000000)
  %108 = getelementptr inbounds nuw float, ptr %8, i64 %11
  store <8 x float> %107, ptr %108, align 4, !alias.scope !10, !noalias !14
  %index.next = add nuw i64 %index, 8
  %109 = icmp eq i64 %index.next, 64
  br i1 %109, label %middle.block, label %vector.body, !llvm.loop !15

middle.block:                                     ; preds = %vector.body
  %110 = add nuw nsw i64 %9, 1
  %exitcond2.not = icmp eq i64 %110, 16
  br i1 %exitcond2.not, label %broadcast_multiply_fusion_wrapped.exit, label %vector.ph, !llvm.loop !18

broadcast_multiply_fusion_wrapped.exit:           ; preds = %middle.block
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 4}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4096}
!5 = !{!6}
!6 = distinct !{!6, !7, !"broadcast_multiply_fusion_wrapped: argument 0"}
!7 = distinct !{!7, !"broadcast_multiply_fusion_wrapped"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"broadcast_multiply_fusion_wrapped: argument 1"}
!10 = !{!11}
!11 = distinct !{!11, !7, !"broadcast_multiply_fusion_wrapped: argument 2"}
!12 = !{!9, !11}
!13 = !{!6, !11}
!14 = !{!6, !9}
!15 = distinct !{!15, !16, !17}
!16 = !{!"llvm.loop.isvectorized", i32 1}
!17 = !{!"llvm.loop.unroll.runtime.disable"}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.unroll.disable"}
