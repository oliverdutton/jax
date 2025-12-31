; ModuleID = '__compute_module_broadcast_add_fusion.3_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.3_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @broadcast_add_fusion.3(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  %7 = load i32, ptr %4, align 4, !invariant.load !3, !alias.scope !6, !noalias !9
  br label %.preheader

.preheader:                                       ; preds = %1, %.preheader
  %8 = phi i64 [ 0, %1 ], [ %41, %.preheader ]
  %.idx = shl nuw nsw i64 %8, 7
  %9 = getelementptr i8, ptr %6, i64 %.idx
  store i32 %7, ptr %9, align 4, !alias.scope !9, !noalias !6
  %10 = getelementptr i8, ptr %9, i64 4
  store i32 %7, ptr %10, align 4, !alias.scope !9, !noalias !6
  %11 = getelementptr i8, ptr %9, i64 8
  store i32 %7, ptr %11, align 4, !alias.scope !9, !noalias !6
  %12 = getelementptr i8, ptr %9, i64 12
  store i32 %7, ptr %12, align 4, !alias.scope !9, !noalias !6
  %13 = getelementptr i8, ptr %9, i64 16
  store i32 %7, ptr %13, align 4, !alias.scope !9, !noalias !6
  %14 = getelementptr i8, ptr %9, i64 20
  store i32 %7, ptr %14, align 4, !alias.scope !9, !noalias !6
  %15 = getelementptr i8, ptr %9, i64 24
  store i32 %7, ptr %15, align 4, !alias.scope !9, !noalias !6
  %16 = getelementptr i8, ptr %9, i64 28
  store i32 %7, ptr %16, align 4, !alias.scope !9, !noalias !6
  %17 = getelementptr i8, ptr %9, i64 32
  store i32 %7, ptr %17, align 4, !alias.scope !9, !noalias !6
  %18 = getelementptr i8, ptr %9, i64 36
  store i32 %7, ptr %18, align 4, !alias.scope !9, !noalias !6
  %19 = getelementptr i8, ptr %9, i64 40
  store i32 %7, ptr %19, align 4, !alias.scope !9, !noalias !6
  %20 = getelementptr i8, ptr %9, i64 44
  store i32 %7, ptr %20, align 4, !alias.scope !9, !noalias !6
  %21 = getelementptr i8, ptr %9, i64 48
  store i32 %7, ptr %21, align 4, !alias.scope !9, !noalias !6
  %22 = getelementptr i8, ptr %9, i64 52
  store i32 %7, ptr %22, align 4, !alias.scope !9, !noalias !6
  %23 = getelementptr i8, ptr %9, i64 56
  store i32 %7, ptr %23, align 4, !alias.scope !9, !noalias !6
  %24 = getelementptr i8, ptr %9, i64 60
  store i32 %7, ptr %24, align 4, !alias.scope !9, !noalias !6
  %25 = getelementptr i8, ptr %9, i64 64
  store i32 %7, ptr %25, align 4, !alias.scope !9, !noalias !6
  %26 = getelementptr i8, ptr %9, i64 68
  store i32 %7, ptr %26, align 4, !alias.scope !9, !noalias !6
  %27 = getelementptr i8, ptr %9, i64 72
  store i32 %7, ptr %27, align 4, !alias.scope !9, !noalias !6
  %28 = getelementptr i8, ptr %9, i64 76
  store i32 %7, ptr %28, align 4, !alias.scope !9, !noalias !6
  %29 = getelementptr i8, ptr %9, i64 80
  store i32 %7, ptr %29, align 4, !alias.scope !9, !noalias !6
  %30 = getelementptr i8, ptr %9, i64 84
  store i32 %7, ptr %30, align 4, !alias.scope !9, !noalias !6
  %31 = getelementptr i8, ptr %9, i64 88
  store i32 %7, ptr %31, align 4, !alias.scope !9, !noalias !6
  %32 = getelementptr i8, ptr %9, i64 92
  store i32 %7, ptr %32, align 4, !alias.scope !9, !noalias !6
  %33 = getelementptr i8, ptr %9, i64 96
  store i32 %7, ptr %33, align 4, !alias.scope !9, !noalias !6
  %34 = getelementptr i8, ptr %9, i64 100
  store i32 %7, ptr %34, align 4, !alias.scope !9, !noalias !6
  %35 = getelementptr i8, ptr %9, i64 104
  store i32 %7, ptr %35, align 4, !alias.scope !9, !noalias !6
  %36 = getelementptr i8, ptr %9, i64 108
  store i32 %7, ptr %36, align 4, !alias.scope !9, !noalias !6
  %37 = getelementptr i8, ptr %9, i64 112
  store i32 %7, ptr %37, align 4, !alias.scope !9, !noalias !6
  %38 = getelementptr i8, ptr %9, i64 116
  store i32 %7, ptr %38, align 4, !alias.scope !9, !noalias !6
  %39 = getelementptr i8, ptr %9, i64 120
  store i32 %7, ptr %39, align 4, !alias.scope !9, !noalias !6
  %40 = getelementptr i8, ptr %9, i64 124
  store i32 %7, ptr %40, align 4, !alias.scope !9, !noalias !6
  %41 = add nuw nsw i64 %8, 1
  %exitcond.not = icmp eq i64 %41, 64
  br i1 %exitcond.not, label %broadcast_add_fusion.3_wrapped.exit, label %.preheader, !llvm.loop !11

broadcast_add_fusion.3_wrapped.exit:              ; preds = %.preheader
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 3}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 8}
!5 = !{i64 8192}
!6 = !{!7}
!7 = distinct !{!7, !8, !"broadcast_add_fusion.3_wrapped: argument 0"}
!8 = distinct !{!8, !"broadcast_add_fusion.3_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"broadcast_add_fusion.3_wrapped: argument 1"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
