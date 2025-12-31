; ModuleID = '__compute_module_broadcast_multiply_fusion_kernel_module'
source_filename = "__compute_module_broadcast_multiply_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @broadcast_multiply_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !4
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  %9 = load float, ptr %6, align 4, !invariant.load !3, !alias.scope !9, !noalias !13
  %broadcast.splatinsert = insertelement <8 x float> poison, float %9, i64 0
  %broadcast.splat = shufflevector <8 x float> %broadcast.splatinsert, <8 x float> poison, <8 x i32> zeroinitializer
  br label %vector.ph

vector.ph:                                        ; preds = %1, %vector.ph
  %10 = phi i64 [ 0, %1 ], [ %37, %vector.ph ]
  %11 = shl nuw nsw i64 %10, 6
  %12 = getelementptr inbounds nuw float, ptr %4, i64 %11
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 64
  %15 = getelementptr inbounds nuw i8, ptr %12, i64 96
  %wide.load = load <8 x float>, ptr %12, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %wide.load3 = load <8 x float>, ptr %13, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %wide.load4 = load <8 x float>, ptr %14, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %wide.load5 = load <8 x float>, ptr %15, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %16 = fmul <8 x float> %broadcast.splat, %wide.load
  %17 = fmul <8 x float> %broadcast.splat, %wide.load3
  %18 = fmul <8 x float> %broadcast.splat, %wide.load4
  %19 = fmul <8 x float> %broadcast.splat, %wide.load5
  %20 = getelementptr inbounds nuw float, ptr %8, i64 %11
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %22 = getelementptr inbounds nuw i8, ptr %20, i64 64
  %23 = getelementptr inbounds nuw i8, ptr %20, i64 96
  store <8 x float> %16, ptr %20, align 4, !alias.scope !11, !noalias !15
  store <8 x float> %17, ptr %21, align 4, !alias.scope !11, !noalias !15
  store <8 x float> %18, ptr %22, align 4, !alias.scope !11, !noalias !15
  store <8 x float> %19, ptr %23, align 4, !alias.scope !11, !noalias !15
  %24 = or disjoint i64 %11, 32
  %25 = getelementptr inbounds nuw float, ptr %4, i64 %24
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 32
  %27 = getelementptr inbounds nuw i8, ptr %25, i64 64
  %28 = getelementptr inbounds nuw i8, ptr %25, i64 96
  %wide.load.1 = load <8 x float>, ptr %25, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %wide.load3.1 = load <8 x float>, ptr %26, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %wide.load4.1 = load <8 x float>, ptr %27, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %wide.load5.1 = load <8 x float>, ptr %28, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %29 = fmul <8 x float> %broadcast.splat, %wide.load.1
  %30 = fmul <8 x float> %broadcast.splat, %wide.load3.1
  %31 = fmul <8 x float> %broadcast.splat, %wide.load4.1
  %32 = fmul <8 x float> %broadcast.splat, %wide.load5.1
  %33 = getelementptr inbounds nuw float, ptr %8, i64 %24
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 32
  %35 = getelementptr inbounds nuw i8, ptr %33, i64 64
  %36 = getelementptr inbounds nuw i8, ptr %33, i64 96
  store <8 x float> %29, ptr %33, align 4, !alias.scope !11, !noalias !15
  store <8 x float> %30, ptr %34, align 4, !alias.scope !11, !noalias !15
  store <8 x float> %31, ptr %35, align 4, !alias.scope !11, !noalias !15
  store <8 x float> %32, ptr %36, align 4, !alias.scope !11, !noalias !15
  %37 = add nuw nsw i64 %10, 1
  %exitcond2.not = icmp eq i64 %37, 64
  br i1 %exitcond2.not, label %broadcast_multiply_fusion_wrapped.exit, label %vector.ph, !llvm.loop !16

broadcast_multiply_fusion_wrapped.exit:           ; preds = %vector.ph
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 16384}
!5 = !{i64 4}
!6 = !{!7}
!7 = distinct !{!7, !8, !"broadcast_multiply_fusion_wrapped: argument 0"}
!8 = distinct !{!8, !"broadcast_multiply_fusion_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"broadcast_multiply_fusion_wrapped: argument 1"}
!11 = !{!12}
!12 = distinct !{!12, !8, !"broadcast_multiply_fusion_wrapped: argument 2"}
!13 = !{!7, !12}
!14 = !{!10, !12}
!15 = !{!7, !10}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.unroll.disable"}
