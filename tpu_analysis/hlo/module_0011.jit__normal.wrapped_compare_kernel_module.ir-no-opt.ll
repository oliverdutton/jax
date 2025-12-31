; ModuleID = '__compute_module_wrapped_compare_kernel_module'
source_filename = "__compute_module_wrapped_compare_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @wrapped_compare(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !4
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !5
  %10 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 0
  %13 = load i64, ptr %12, align 4, !invariant.load !3
  %14 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 1
  %15 = load i64, ptr %14, align 4, !invariant.load !3
  %16 = getelementptr inbounds %kernel_dim3, ptr %11, i32 0, i32 2
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  call void @wrapped_compare_wrapped(ptr %5, ptr %7, ptr %9, i64 %13, i64 %15, i64 %17)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @wrapped_compare_wrapped(ptr noalias align 64 dereferenceable(4) %0, ptr noalias align 64 dereferenceable(4) %1, ptr noalias align 64 dereferenceable(1) %2, i64 %3, i64 %4, i64 %5) #1 {
  %7 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %8 = load i32, ptr %7, align 4, !invariant.load !3
  %9 = getelementptr inbounds [1 x i32], ptr %1, i32 0, i32 0
  %10 = load i32, ptr %9, align 4, !invariant.load !3
  %11 = icmp slt i32 %8, %10
  %12 = zext i1 %11 to i8
  %13 = getelementptr inbounds [1 x i8], ptr %2, i32 0, i32 0
  store i8 %12, ptr %13, align 1
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 8}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 1}
