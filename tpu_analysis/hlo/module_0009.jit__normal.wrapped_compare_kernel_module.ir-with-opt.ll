; ModuleID = '__compute_module_wrapped_compare_kernel_module'
source_filename = "__compute_module_wrapped_compare_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @wrapped_compare(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !9)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !11)
  %9 = load i32, ptr %4, align 4, !invariant.load !3, !alias.scope !6, !noalias !13
  %10 = load i32, ptr %6, align 4, !invariant.load !3, !alias.scope !9, !noalias !14
  %11 = icmp slt i32 %9, %10
  %12 = zext i1 %11 to i8
  store i8 %12, ptr %8, align 1, !alias.scope !11, !noalias !15
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 8}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 1}
!6 = !{!7}
!7 = distinct !{!7, !8, !"wrapped_compare_wrapped: argument 0"}
!8 = distinct !{!8, !"wrapped_compare_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"wrapped_compare_wrapped: argument 1"}
!11 = !{!12}
!12 = distinct !{!12, !8, !"wrapped_compare_wrapped: argument 2"}
!13 = !{!10, !12}
!14 = !{!7, !12}
!15 = !{!7, !10}
