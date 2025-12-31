; ModuleID = '__compute_module_broadcast_add_fusion.2_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.2_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @broadcast_add_fusion.2(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 4
  %8 = load i32, ptr %7, align 4, !invariant.load !3, !alias.scope !6, !noalias !9
  %broadcast.splatinsert3 = insertelement <8 x i32> poison, i32 %8, i64 0
  %broadcast.splat4 = shufflevector <8 x i32> %broadcast.splatinsert3, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph

vector.ph:                                        ; preds = %1, %vector.ph
  %9 = phi i64 [ 0, %1 ], [ %43, %vector.ph ]
  %10 = shl nuw nsw i64 %9, 6
  %broadcast.splatinsert = insertelement <8 x i64> poison, i64 %10, i64 0
  %broadcast.splat = shufflevector <8 x i64> %broadcast.splatinsert, <8 x i64> poison, <8 x i32> zeroinitializer
  %.idx = shl nuw nsw i64 %9, 8
  %11 = getelementptr i8, ptr %6, i64 %.idx
  %12 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %13 = or disjoint <8 x i32> %12, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %14 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %15 = or disjoint <8 x i32> %14, <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %16 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %17 = or disjoint <8 x i32> %16, <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %18 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %19 = or disjoint <8 x i32> %18, <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %20 = add <8 x i32> %broadcast.splat4, %13
  %21 = add <8 x i32> %broadcast.splat4, %15
  %22 = add <8 x i32> %broadcast.splat4, %17
  %23 = add <8 x i32> %broadcast.splat4, %19
  %24 = getelementptr i8, ptr %11, i64 32
  %25 = getelementptr i8, ptr %11, i64 64
  %26 = getelementptr i8, ptr %11, i64 96
  store <8 x i32> %20, ptr %11, align 4, !alias.scope !9, !noalias !6
  store <8 x i32> %21, ptr %24, align 4, !alias.scope !9, !noalias !6
  store <8 x i32> %22, ptr %25, align 4, !alias.scope !9, !noalias !6
  store <8 x i32> %23, ptr %26, align 4, !alias.scope !9, !noalias !6
  %27 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %28 = or disjoint <8 x i32> %27, <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39>
  %29 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %30 = or disjoint <8 x i32> %29, <i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>
  %31 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %32 = or disjoint <8 x i32> %31, <i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55>
  %33 = trunc <8 x i64> %broadcast.splat to <8 x i32>
  %34 = or disjoint <8 x i32> %33, <i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %35 = add <8 x i32> %broadcast.splat4, %28
  %36 = add <8 x i32> %broadcast.splat4, %30
  %37 = add <8 x i32> %broadcast.splat4, %32
  %38 = add <8 x i32> %broadcast.splat4, %34
  %39 = getelementptr i8, ptr %11, i64 128
  %40 = getelementptr i8, ptr %11, i64 160
  %41 = getelementptr i8, ptr %11, i64 192
  %42 = getelementptr i8, ptr %11, i64 224
  store <8 x i32> %35, ptr %39, align 4, !alias.scope !9, !noalias !6
  store <8 x i32> %36, ptr %40, align 4, !alias.scope !9, !noalias !6
  store <8 x i32> %37, ptr %41, align 4, !alias.scope !9, !noalias !6
  store <8 x i32> %38, ptr %42, align 4, !alias.scope !9, !noalias !6
  %43 = add nuw nsw i64 %9, 1
  %exitcond2.not = icmp eq i64 %43, 16
  br i1 %exitcond2.not, label %broadcast_add_fusion.2_wrapped.exit, label %vector.ph, !llvm.loop !11

broadcast_add_fusion.2_wrapped.exit:              ; preds = %vector.ph
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 2}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{i64 4096}
!6 = !{!7}
!7 = distinct !{!7, !8, !"broadcast_add_fusion.2_wrapped: argument 0"}
!8 = distinct !{!8, !"broadcast_add_fusion.2_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"broadcast_add_fusion.2_wrapped: argument 1"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
