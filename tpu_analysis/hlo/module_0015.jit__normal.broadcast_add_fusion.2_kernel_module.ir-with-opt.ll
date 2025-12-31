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
  br label %9

9:                                                ; preds = %1, %9
  %10 = phi i64 [ 0, %1 ], [ %139, %9 ]
  %11 = shl nuw nsw i64 %10, 5
  %.idx = shl nuw nsw i64 %10, 7
  %12 = getelementptr i8, ptr %6, i64 %.idx
  %13 = trunc nuw nsw i64 %11 to i32
  %14 = add i32 %8, %13
  store i32 %14, ptr %12, align 4, !alias.scope !9, !noalias !6
  %15 = trunc i64 %11 to i32
  %16 = or disjoint i32 %15, 1
  %17 = add i32 %8, %16
  %18 = getelementptr i8, ptr %12, i64 4
  store i32 %17, ptr %18, align 4, !alias.scope !9, !noalias !6
  %19 = trunc i64 %11 to i32
  %20 = or disjoint i32 %19, 2
  %21 = add i32 %8, %20
  %22 = getelementptr i8, ptr %12, i64 8
  store i32 %21, ptr %22, align 4, !alias.scope !9, !noalias !6
  %23 = trunc i64 %11 to i32
  %24 = or disjoint i32 %23, 3
  %25 = add i32 %8, %24
  %26 = getelementptr i8, ptr %12, i64 12
  store i32 %25, ptr %26, align 4, !alias.scope !9, !noalias !6
  %27 = trunc i64 %11 to i32
  %28 = or disjoint i32 %27, 4
  %29 = add i32 %8, %28
  %30 = getelementptr i8, ptr %12, i64 16
  store i32 %29, ptr %30, align 4, !alias.scope !9, !noalias !6
  %31 = trunc i64 %11 to i32
  %32 = or disjoint i32 %31, 5
  %33 = add i32 %8, %32
  %34 = getelementptr i8, ptr %12, i64 20
  store i32 %33, ptr %34, align 4, !alias.scope !9, !noalias !6
  %35 = trunc i64 %11 to i32
  %36 = or disjoint i32 %35, 6
  %37 = add i32 %8, %36
  %38 = getelementptr i8, ptr %12, i64 24
  store i32 %37, ptr %38, align 4, !alias.scope !9, !noalias !6
  %39 = trunc i64 %11 to i32
  %40 = or disjoint i32 %39, 7
  %41 = add i32 %8, %40
  %42 = getelementptr i8, ptr %12, i64 28
  store i32 %41, ptr %42, align 4, !alias.scope !9, !noalias !6
  %43 = trunc i64 %11 to i32
  %44 = or disjoint i32 %43, 8
  %45 = add i32 %8, %44
  %46 = getelementptr i8, ptr %12, i64 32
  store i32 %45, ptr %46, align 4, !alias.scope !9, !noalias !6
  %47 = trunc i64 %11 to i32
  %48 = or disjoint i32 %47, 9
  %49 = add i32 %8, %48
  %50 = getelementptr i8, ptr %12, i64 36
  store i32 %49, ptr %50, align 4, !alias.scope !9, !noalias !6
  %51 = trunc i64 %11 to i32
  %52 = or disjoint i32 %51, 10
  %53 = add i32 %8, %52
  %54 = getelementptr i8, ptr %12, i64 40
  store i32 %53, ptr %54, align 4, !alias.scope !9, !noalias !6
  %55 = trunc i64 %11 to i32
  %56 = or disjoint i32 %55, 11
  %57 = add i32 %8, %56
  %58 = getelementptr i8, ptr %12, i64 44
  store i32 %57, ptr %58, align 4, !alias.scope !9, !noalias !6
  %59 = trunc i64 %11 to i32
  %60 = or disjoint i32 %59, 12
  %61 = add i32 %8, %60
  %62 = getelementptr i8, ptr %12, i64 48
  store i32 %61, ptr %62, align 4, !alias.scope !9, !noalias !6
  %63 = trunc i64 %11 to i32
  %64 = or disjoint i32 %63, 13
  %65 = add i32 %8, %64
  %66 = getelementptr i8, ptr %12, i64 52
  store i32 %65, ptr %66, align 4, !alias.scope !9, !noalias !6
  %67 = trunc i64 %11 to i32
  %68 = or disjoint i32 %67, 14
  %69 = add i32 %8, %68
  %70 = getelementptr i8, ptr %12, i64 56
  store i32 %69, ptr %70, align 4, !alias.scope !9, !noalias !6
  %71 = trunc i64 %11 to i32
  %72 = or disjoint i32 %71, 15
  %73 = add i32 %8, %72
  %74 = getelementptr i8, ptr %12, i64 60
  store i32 %73, ptr %74, align 4, !alias.scope !9, !noalias !6
  %75 = trunc i64 %11 to i32
  %76 = or disjoint i32 %75, 16
  %77 = add i32 %8, %76
  %78 = getelementptr i8, ptr %12, i64 64
  store i32 %77, ptr %78, align 4, !alias.scope !9, !noalias !6
  %79 = trunc i64 %11 to i32
  %80 = or disjoint i32 %79, 17
  %81 = add i32 %8, %80
  %82 = getelementptr i8, ptr %12, i64 68
  store i32 %81, ptr %82, align 4, !alias.scope !9, !noalias !6
  %83 = trunc i64 %11 to i32
  %84 = or disjoint i32 %83, 18
  %85 = add i32 %8, %84
  %86 = getelementptr i8, ptr %12, i64 72
  store i32 %85, ptr %86, align 4, !alias.scope !9, !noalias !6
  %87 = trunc i64 %11 to i32
  %88 = or disjoint i32 %87, 19
  %89 = add i32 %8, %88
  %90 = getelementptr i8, ptr %12, i64 76
  store i32 %89, ptr %90, align 4, !alias.scope !9, !noalias !6
  %91 = trunc i64 %11 to i32
  %92 = or disjoint i32 %91, 20
  %93 = add i32 %8, %92
  %94 = getelementptr i8, ptr %12, i64 80
  store i32 %93, ptr %94, align 4, !alias.scope !9, !noalias !6
  %95 = trunc i64 %11 to i32
  %96 = or disjoint i32 %95, 21
  %97 = add i32 %8, %96
  %98 = getelementptr i8, ptr %12, i64 84
  store i32 %97, ptr %98, align 4, !alias.scope !9, !noalias !6
  %99 = trunc i64 %11 to i32
  %100 = or disjoint i32 %99, 22
  %101 = add i32 %8, %100
  %102 = getelementptr i8, ptr %12, i64 88
  store i32 %101, ptr %102, align 4, !alias.scope !9, !noalias !6
  %103 = trunc i64 %11 to i32
  %104 = or disjoint i32 %103, 23
  %105 = add i32 %8, %104
  %106 = getelementptr i8, ptr %12, i64 92
  store i32 %105, ptr %106, align 4, !alias.scope !9, !noalias !6
  %107 = trunc i64 %11 to i32
  %108 = or disjoint i32 %107, 24
  %109 = add i32 %8, %108
  %110 = getelementptr i8, ptr %12, i64 96
  store i32 %109, ptr %110, align 4, !alias.scope !9, !noalias !6
  %111 = trunc i64 %11 to i32
  %112 = or disjoint i32 %111, 25
  %113 = add i32 %8, %112
  %114 = getelementptr i8, ptr %12, i64 100
  store i32 %113, ptr %114, align 4, !alias.scope !9, !noalias !6
  %115 = trunc i64 %11 to i32
  %116 = or disjoint i32 %115, 26
  %117 = add i32 %8, %116
  %118 = getelementptr i8, ptr %12, i64 104
  store i32 %117, ptr %118, align 4, !alias.scope !9, !noalias !6
  %119 = trunc i64 %11 to i32
  %120 = or disjoint i32 %119, 27
  %121 = add i32 %8, %120
  %122 = getelementptr i8, ptr %12, i64 108
  store i32 %121, ptr %122, align 4, !alias.scope !9, !noalias !6
  %123 = trunc i64 %11 to i32
  %124 = or disjoint i32 %123, 28
  %125 = add i32 %8, %124
  %126 = getelementptr i8, ptr %12, i64 112
  store i32 %125, ptr %126, align 4, !alias.scope !9, !noalias !6
  %127 = trunc i64 %11 to i32
  %128 = or disjoint i32 %127, 29
  %129 = add i32 %8, %128
  %130 = getelementptr i8, ptr %12, i64 116
  store i32 %129, ptr %130, align 4, !alias.scope !9, !noalias !6
  %131 = trunc i64 %11 to i32
  %132 = or disjoint i32 %131, 30
  %133 = add i32 %8, %132
  %134 = getelementptr i8, ptr %12, i64 120
  store i32 %133, ptr %134, align 4, !alias.scope !9, !noalias !6
  %135 = trunc i64 %11 to i32
  %136 = or disjoint i32 %135, 31
  %137 = add i32 %8, %136
  %138 = getelementptr i8, ptr %12, i64 124
  store i32 %137, ptr %138, align 4, !alias.scope !9, !noalias !6
  %139 = add nuw nsw i64 %10, 1
  %exitcond.not = icmp eq i64 %139, 64
  br i1 %exitcond.not, label %broadcast_add_fusion.2_wrapped.exit, label %9, !llvm.loop !11

broadcast_add_fusion.2_wrapped.exit:              ; preds = %9
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
!5 = !{i64 8192}
!6 = !{!7}
!7 = distinct !{!7, !8, !"broadcast_add_fusion.2_wrapped: argument 0"}
!8 = distinct !{!8, !"broadcast_add_fusion.2_wrapped"}
!9 = !{!10}
!10 = distinct !{!10, !8, !"broadcast_add_fusion.2_wrapped: argument 1"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
