; ModuleID = '__compute_module_bitcast_concatenate_fusion_kernel_module'
source_filename = "__compute_module_bitcast_concatenate_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @bitcast_concatenate_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
.preheader.preheader:
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %2 = load ptr, ptr %1, align 8, !invariant.load !3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3, !dereferenceable !4
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  %8 = load i32, ptr %5, align 4, !invariant.load !3, !alias.scope !9, !noalias !13
  store i32 %8, ptr %7, align 4, !alias.scope !11, !noalias !14
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 4
  %10 = load i32, ptr %9, align 4, !invariant.load !3, !alias.scope !9, !noalias !13
  %11 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i32 %10, ptr %11, align 4, !alias.scope !11, !noalias !14
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %13 = load i32, ptr %12, align 4, !invariant.load !3, !alias.scope !9, !noalias !13
  %14 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store i32 %13, ptr %14, align 4, !alias.scope !11, !noalias !14
  %15 = load i32, ptr %3, align 4, !invariant.load !3, !alias.scope !6, !noalias !15
  %16 = getelementptr inbounds nuw i8, ptr %7, i64 4
  store i32 %15, ptr %16, align 4, !alias.scope !11, !noalias !14
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 4
  %18 = load i32, ptr %17, align 4, !invariant.load !3, !alias.scope !6, !noalias !15
  %19 = getelementptr inbounds nuw i8, ptr %7, i64 12
  store i32 %18, ptr %19, align 4, !alias.scope !11, !noalias !14
  %20 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %21 = load i32, ptr %20, align 4, !invariant.load !3, !alias.scope !6, !noalias !15
  %22 = getelementptr inbounds nuw i8, ptr %7, i64 20
  store i32 %21, ptr %22, align 4, !alias.scope !11, !noalias !14
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 1}
!2 = !{!"xla_cpu_emitter__concatenate_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 12}
!5 = !{i64 24}
!6 = !{!7}
!7 = distinct !{!7, !8, !"bitcast_concatenate_fusion_wrapped: argument 0"}
!8 = distinct !{!8, !"bitcast_concatenate_fusion_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"bitcast_concatenate_fusion_wrapped: argument 1"}
!11 = !{!12}
!12 = distinct !{!12, !8, !"bitcast_concatenate_fusion_wrapped: argument 2"}
!13 = !{!7, !12}
!14 = !{!7, !10}
!15 = !{!10, !12}
