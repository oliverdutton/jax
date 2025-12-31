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
  br label %10

10:                                               ; preds = %1, %10
  %11 = phi i64 [ 0, %1 ], [ %172, %10 ]
  %12 = shl nuw nsw i64 %11, 5
  %13 = getelementptr inbounds nuw float, ptr %4, i64 %12
  %14 = load float, ptr %13, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %15 = fmul float %9, %14
  %16 = getelementptr inbounds nuw float, ptr %8, i64 %12
  store float %15, ptr %16, align 4, !alias.scope !11, !noalias !15
  %17 = or disjoint i64 %12, 1
  %18 = getelementptr inbounds nuw float, ptr %4, i64 %17
  %19 = load float, ptr %18, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %20 = fmul float %9, %19
  %21 = getelementptr inbounds nuw float, ptr %8, i64 %17
  store float %20, ptr %21, align 4, !alias.scope !11, !noalias !15
  %22 = or disjoint i64 %12, 2
  %23 = getelementptr inbounds nuw float, ptr %4, i64 %22
  %24 = load float, ptr %23, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %25 = fmul float %9, %24
  %26 = getelementptr inbounds nuw float, ptr %8, i64 %22
  store float %25, ptr %26, align 4, !alias.scope !11, !noalias !15
  %27 = or disjoint i64 %12, 3
  %28 = getelementptr inbounds nuw float, ptr %4, i64 %27
  %29 = load float, ptr %28, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %30 = fmul float %9, %29
  %31 = getelementptr inbounds nuw float, ptr %8, i64 %27
  store float %30, ptr %31, align 4, !alias.scope !11, !noalias !15
  %32 = or disjoint i64 %12, 4
  %33 = getelementptr inbounds nuw float, ptr %4, i64 %32
  %34 = load float, ptr %33, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %35 = fmul float %9, %34
  %36 = getelementptr inbounds nuw float, ptr %8, i64 %32
  store float %35, ptr %36, align 4, !alias.scope !11, !noalias !15
  %37 = or disjoint i64 %12, 5
  %38 = getelementptr inbounds nuw float, ptr %4, i64 %37
  %39 = load float, ptr %38, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %40 = fmul float %9, %39
  %41 = getelementptr inbounds nuw float, ptr %8, i64 %37
  store float %40, ptr %41, align 4, !alias.scope !11, !noalias !15
  %42 = or disjoint i64 %12, 6
  %43 = getelementptr inbounds nuw float, ptr %4, i64 %42
  %44 = load float, ptr %43, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %45 = fmul float %9, %44
  %46 = getelementptr inbounds nuw float, ptr %8, i64 %42
  store float %45, ptr %46, align 4, !alias.scope !11, !noalias !15
  %47 = or disjoint i64 %12, 7
  %48 = getelementptr inbounds nuw float, ptr %4, i64 %47
  %49 = load float, ptr %48, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %50 = fmul float %9, %49
  %51 = getelementptr inbounds nuw float, ptr %8, i64 %47
  store float %50, ptr %51, align 4, !alias.scope !11, !noalias !15
  %52 = or disjoint i64 %12, 8
  %53 = getelementptr inbounds nuw float, ptr %4, i64 %52
  %54 = load float, ptr %53, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %55 = fmul float %9, %54
  %56 = getelementptr inbounds nuw float, ptr %8, i64 %52
  store float %55, ptr %56, align 4, !alias.scope !11, !noalias !15
  %57 = or disjoint i64 %12, 9
  %58 = getelementptr inbounds nuw float, ptr %4, i64 %57
  %59 = load float, ptr %58, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %60 = fmul float %9, %59
  %61 = getelementptr inbounds nuw float, ptr %8, i64 %57
  store float %60, ptr %61, align 4, !alias.scope !11, !noalias !15
  %62 = or disjoint i64 %12, 10
  %63 = getelementptr inbounds nuw float, ptr %4, i64 %62
  %64 = load float, ptr %63, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %65 = fmul float %9, %64
  %66 = getelementptr inbounds nuw float, ptr %8, i64 %62
  store float %65, ptr %66, align 4, !alias.scope !11, !noalias !15
  %67 = or disjoint i64 %12, 11
  %68 = getelementptr inbounds nuw float, ptr %4, i64 %67
  %69 = load float, ptr %68, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %70 = fmul float %9, %69
  %71 = getelementptr inbounds nuw float, ptr %8, i64 %67
  store float %70, ptr %71, align 4, !alias.scope !11, !noalias !15
  %72 = or disjoint i64 %12, 12
  %73 = getelementptr inbounds nuw float, ptr %4, i64 %72
  %74 = load float, ptr %73, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %75 = fmul float %9, %74
  %76 = getelementptr inbounds nuw float, ptr %8, i64 %72
  store float %75, ptr %76, align 4, !alias.scope !11, !noalias !15
  %77 = or disjoint i64 %12, 13
  %78 = getelementptr inbounds nuw float, ptr %4, i64 %77
  %79 = load float, ptr %78, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %80 = fmul float %9, %79
  %81 = getelementptr inbounds nuw float, ptr %8, i64 %77
  store float %80, ptr %81, align 4, !alias.scope !11, !noalias !15
  %82 = or disjoint i64 %12, 14
  %83 = getelementptr inbounds nuw float, ptr %4, i64 %82
  %84 = load float, ptr %83, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %85 = fmul float %9, %84
  %86 = getelementptr inbounds nuw float, ptr %8, i64 %82
  store float %85, ptr %86, align 4, !alias.scope !11, !noalias !15
  %87 = or disjoint i64 %12, 15
  %88 = getelementptr inbounds nuw float, ptr %4, i64 %87
  %89 = load float, ptr %88, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %90 = fmul float %9, %89
  %91 = getelementptr inbounds nuw float, ptr %8, i64 %87
  store float %90, ptr %91, align 4, !alias.scope !11, !noalias !15
  %92 = or disjoint i64 %12, 16
  %93 = getelementptr inbounds nuw float, ptr %4, i64 %92
  %94 = load float, ptr %93, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %95 = fmul float %9, %94
  %96 = getelementptr inbounds nuw float, ptr %8, i64 %92
  store float %95, ptr %96, align 4, !alias.scope !11, !noalias !15
  %97 = or disjoint i64 %12, 17
  %98 = getelementptr inbounds nuw float, ptr %4, i64 %97
  %99 = load float, ptr %98, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %100 = fmul float %9, %99
  %101 = getelementptr inbounds nuw float, ptr %8, i64 %97
  store float %100, ptr %101, align 4, !alias.scope !11, !noalias !15
  %102 = or disjoint i64 %12, 18
  %103 = getelementptr inbounds nuw float, ptr %4, i64 %102
  %104 = load float, ptr %103, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %105 = fmul float %9, %104
  %106 = getelementptr inbounds nuw float, ptr %8, i64 %102
  store float %105, ptr %106, align 4, !alias.scope !11, !noalias !15
  %107 = or disjoint i64 %12, 19
  %108 = getelementptr inbounds nuw float, ptr %4, i64 %107
  %109 = load float, ptr %108, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %110 = fmul float %9, %109
  %111 = getelementptr inbounds nuw float, ptr %8, i64 %107
  store float %110, ptr %111, align 4, !alias.scope !11, !noalias !15
  %112 = or disjoint i64 %12, 20
  %113 = getelementptr inbounds nuw float, ptr %4, i64 %112
  %114 = load float, ptr %113, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %115 = fmul float %9, %114
  %116 = getelementptr inbounds nuw float, ptr %8, i64 %112
  store float %115, ptr %116, align 4, !alias.scope !11, !noalias !15
  %117 = or disjoint i64 %12, 21
  %118 = getelementptr inbounds nuw float, ptr %4, i64 %117
  %119 = load float, ptr %118, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %120 = fmul float %9, %119
  %121 = getelementptr inbounds nuw float, ptr %8, i64 %117
  store float %120, ptr %121, align 4, !alias.scope !11, !noalias !15
  %122 = or disjoint i64 %12, 22
  %123 = getelementptr inbounds nuw float, ptr %4, i64 %122
  %124 = load float, ptr %123, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %125 = fmul float %9, %124
  %126 = getelementptr inbounds nuw float, ptr %8, i64 %122
  store float %125, ptr %126, align 4, !alias.scope !11, !noalias !15
  %127 = or disjoint i64 %12, 23
  %128 = getelementptr inbounds nuw float, ptr %4, i64 %127
  %129 = load float, ptr %128, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %130 = fmul float %9, %129
  %131 = getelementptr inbounds nuw float, ptr %8, i64 %127
  store float %130, ptr %131, align 4, !alias.scope !11, !noalias !15
  %132 = or disjoint i64 %12, 24
  %133 = getelementptr inbounds nuw float, ptr %4, i64 %132
  %134 = load float, ptr %133, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %135 = fmul float %9, %134
  %136 = getelementptr inbounds nuw float, ptr %8, i64 %132
  store float %135, ptr %136, align 4, !alias.scope !11, !noalias !15
  %137 = or disjoint i64 %12, 25
  %138 = getelementptr inbounds nuw float, ptr %4, i64 %137
  %139 = load float, ptr %138, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %140 = fmul float %9, %139
  %141 = getelementptr inbounds nuw float, ptr %8, i64 %137
  store float %140, ptr %141, align 4, !alias.scope !11, !noalias !15
  %142 = or disjoint i64 %12, 26
  %143 = getelementptr inbounds nuw float, ptr %4, i64 %142
  %144 = load float, ptr %143, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %145 = fmul float %9, %144
  %146 = getelementptr inbounds nuw float, ptr %8, i64 %142
  store float %145, ptr %146, align 4, !alias.scope !11, !noalias !15
  %147 = or disjoint i64 %12, 27
  %148 = getelementptr inbounds nuw float, ptr %4, i64 %147
  %149 = load float, ptr %148, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %150 = fmul float %9, %149
  %151 = getelementptr inbounds nuw float, ptr %8, i64 %147
  store float %150, ptr %151, align 4, !alias.scope !11, !noalias !15
  %152 = or disjoint i64 %12, 28
  %153 = getelementptr inbounds nuw float, ptr %4, i64 %152
  %154 = load float, ptr %153, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %155 = fmul float %9, %154
  %156 = getelementptr inbounds nuw float, ptr %8, i64 %152
  store float %155, ptr %156, align 4, !alias.scope !11, !noalias !15
  %157 = or disjoint i64 %12, 29
  %158 = getelementptr inbounds nuw float, ptr %4, i64 %157
  %159 = load float, ptr %158, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %160 = fmul float %9, %159
  %161 = getelementptr inbounds nuw float, ptr %8, i64 %157
  store float %160, ptr %161, align 4, !alias.scope !11, !noalias !15
  %162 = or disjoint i64 %12, 30
  %163 = getelementptr inbounds nuw float, ptr %4, i64 %162
  %164 = load float, ptr %163, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %165 = fmul float %9, %164
  %166 = getelementptr inbounds nuw float, ptr %8, i64 %162
  store float %165, ptr %166, align 4, !alias.scope !11, !noalias !15
  %167 = or disjoint i64 %12, 31
  %168 = getelementptr inbounds nuw float, ptr %4, i64 %167
  %169 = load float, ptr %168, align 4, !invariant.load !3, !alias.scope !6, !noalias !14
  %170 = fmul float %9, %169
  %171 = getelementptr inbounds nuw float, ptr %8, i64 %167
  store float %170, ptr %171, align 4, !alias.scope !11, !noalias !15
  %172 = add nuw nsw i64 %11, 1
  %exitcond.not = icmp eq i64 %172, 64
  br i1 %exitcond.not, label %broadcast_multiply_fusion_wrapped.exit, label %10, !llvm.loop !16

broadcast_multiply_fusion_wrapped.exit:           ; preds = %10
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
!4 = !{i64 8192}
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
