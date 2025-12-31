; ModuleID = '__compute_module_broadcast_add_fusion.1_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion.1_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @broadcast_add_fusion.1(ptr %0) #0 {
  %2 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8, !invariant.load !3, !dereferenceable !4
  %6 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 1, i32 0
  %7 = load ptr, ptr %6, align 8, !invariant.load !3, !dereferenceable !5
  %8 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 2, i32 0
  %9 = load ptr, ptr %8, align 8, !invariant.load !3, !dereferenceable !6
  %10 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 3, i32 0
  %11 = load ptr, ptr %10, align 8, !invariant.load !3, !dereferenceable !6
  %12 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 4, i32 0
  %13 = load ptr, ptr %12, align 8, !invariant.load !3, !dereferenceable !6
  %14 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr inbounds %kernel_dim3, ptr %15, i32 0, i32 0
  %17 = load i64, ptr %16, align 4, !invariant.load !3
  %18 = getelementptr inbounds %kernel_dim3, ptr %15, i32 0, i32 1
  %19 = load i64, ptr %18, align 4, !invariant.load !3
  %20 = getelementptr inbounds %kernel_dim3, ptr %15, i32 0, i32 2
  %21 = load i64, ptr %20, align 4, !invariant.load !3
  call void @broadcast_add_fusion.1_wrapped(ptr %5, ptr %7, ptr %9, ptr %11, ptr %13, i64 %17, i64 %19, i64 %21)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @broadcast_add_fusion.1_wrapped(ptr noalias align 64 dereferenceable(4) %0, ptr noalias align 64 dereferenceable(16) %1, ptr noalias align 64 dereferenceable(8192) %2, ptr noalias align 64 dereferenceable(8192) %3, ptr noalias align 64 dereferenceable(8192) %4, i64 %5, i64 %6, i64 %7) #1 {
  %9 = icmp sge i64 %5, 0
  %10 = icmp sle i64 %5, 1
  %11 = and i1 %9, %10
  br i1 %11, label %12, label %74

12:                                               ; preds = %8
  %13 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 0
  %14 = load i32, ptr %13, align 4, !invariant.load !3
  %15 = sub i32 32, %14
  %16 = icmp ult i32 %14, 32
  %17 = icmp ult i32 %15, 32
  %18 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 1
  %19 = load i32, ptr %18, align 4, !invariant.load !3
  %20 = sub i32 32, %19
  %21 = icmp ult i32 %19, 32
  %22 = icmp ult i32 %20, 32
  %23 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 2
  %24 = load i32, ptr %23, align 4, !invariant.load !3
  %25 = sub i32 32, %24
  %26 = icmp ult i32 %24, 32
  %27 = icmp ult i32 %25, 32
  %28 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %29 = load i32, ptr %28, align 4, !invariant.load !3
  %30 = mul nsw i64 %5, 1024
  br label %31

31:                                               ; preds = %71, %12
  %32 = phi i64 [ %72, %71 ], [ 0, %12 ]
  %33 = icmp slt i64 %32, 32
  br i1 %33, label %34, label %73

34:                                               ; preds = %31
  %35 = mul nsw i64 %32, 32
  %36 = add nsw i64 %30, %35
  br label %37

37:                                               ; preds = %40, %34
  %38 = phi i64 [ %70, %40 ], [ 0, %34 ]
  %39 = icmp slt i64 %38, 32
  br i1 %39, label %40, label %71

40:                                               ; preds = %37
  %41 = add nsw i64 %36, %38
  %42 = getelementptr inbounds [2048 x i32], ptr %2, i32 0, i64 %41
  %43 = load i32, ptr %42, align 4, !invariant.load !3
  %44 = shl i32 %43, %14
  %45 = select i1 %16, i32 %44, i32 0
  %46 = lshr i32 %43, %15
  %47 = select i1 %17, i32 %46, i32 0
  %48 = getelementptr inbounds [2048 x i32], ptr %3, i32 0, i64 %41
  %49 = load i32, ptr %48, align 4, !invariant.load !3
  %50 = or i32 %45, %47
  %51 = add i32 %49, %43
  %52 = xor i32 %51, %50
  %53 = shl i32 %52, %19
  %54 = select i1 %21, i32 %53, i32 0
  %55 = lshr i32 %52, %20
  %56 = select i1 %22, i32 %55, i32 0
  %57 = or i32 %54, %56
  %58 = add i32 %51, %52
  %59 = xor i32 %58, %57
  %60 = shl i32 %59, %24
  %61 = select i1 %26, i32 %60, i32 0
  %62 = lshr i32 %59, %25
  %63 = select i1 %27, i32 %62, i32 0
  %64 = or i32 %61, %63
  %65 = add i32 %58, %59
  %66 = xor i32 %65, %64
  %67 = add i32 %65, %66
  %68 = add i32 %67, %29
  %69 = getelementptr inbounds [2048 x i32], ptr %4, i32 0, i64 %41
  store i32 %68, ptr %69, align 4
  %70 = add i64 %38, 1
  br label %37

71:                                               ; preds = %37
  %72 = add i64 %32, 1
  br label %31, !llvm.loop !7

73:                                               ; preds = %31
  br label %74

74:                                               ; preds = %73, %8
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 1}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 16}
!6 = !{i64 8192}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.disable"}
