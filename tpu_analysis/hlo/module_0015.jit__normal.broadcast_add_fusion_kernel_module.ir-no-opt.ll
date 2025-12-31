; ModuleID = '__compute_module_broadcast_add_fusion_kernel_module'
source_filename = "__compute_module_broadcast_add_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelArg = type { ptr, i64 }
%kernel_dim3 = type { i64, i64, i64 }

; Function Attrs: uwtable
define ptr @broadcast_add_fusion(ptr %0) #0 {
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
  %13 = load ptr, ptr %12, align 8, !invariant.load !3, !dereferenceable !4
  %14 = getelementptr inbounds %XLA_CPU_KernelArg, ptr %3, i32 5, i32 0
  %15 = load ptr, ptr %14, align 8, !invariant.load !3, !dereferenceable !6
  %16 = getelementptr inbounds %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %17 = load ptr, ptr %16, align 8
  %18 = getelementptr inbounds %kernel_dim3, ptr %17, i32 0, i32 0
  %19 = load i64, ptr %18, align 4, !invariant.load !3
  %20 = getelementptr inbounds %kernel_dim3, ptr %17, i32 0, i32 1
  %21 = load i64, ptr %20, align 4, !invariant.load !3
  %22 = getelementptr inbounds %kernel_dim3, ptr %17, i32 0, i32 2
  %23 = load i64, ptr %22, align 4, !invariant.load !3
  call void @broadcast_add_fusion_wrapped(ptr %5, ptr %7, ptr %9, ptr %11, ptr %13, ptr %15, i64 %19, i64 %21, i64 %23)
  ret ptr null
}

; Function Attrs: alwaysinline
define internal void @broadcast_add_fusion_wrapped(ptr noalias align 64 dereferenceable(4) %0, ptr noalias align 64 dereferenceable(16) %1, ptr noalias align 64 dereferenceable(8192) %2, ptr noalias align 64 dereferenceable(8192) %3, ptr noalias align 64 dereferenceable(4) %4, ptr noalias align 64 dereferenceable(8192) %5, i64 %6, i64 %7, i64 %8) #1 {
  %10 = icmp sge i64 %6, 0
  %11 = icmp sle i64 %6, 1
  %12 = and i1 %10, %11
  br i1 %12, label %13, label %90

13:                                               ; preds = %9
  %14 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 0
  %15 = load i32, ptr %14, align 4, !invariant.load !3
  %16 = sub i32 32, %15
  %17 = icmp ult i32 %15, 32
  %18 = icmp ult i32 %16, 32
  %19 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 1
  %20 = load i32, ptr %19, align 4, !invariant.load !3
  %21 = sub i32 32, %20
  %22 = icmp ult i32 %20, 32
  %23 = icmp ult i32 %21, 32
  %24 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 2
  %25 = load i32, ptr %24, align 4, !invariant.load !3
  %26 = sub i32 32, %25
  %27 = icmp ult i32 %25, 32
  %28 = icmp ult i32 %26, 32
  %29 = getelementptr inbounds [4 x i32], ptr %1, i32 0, i32 3
  %30 = load i32, ptr %29, align 4, !invariant.load !3
  %31 = sub i32 32, %30
  %32 = icmp ult i32 %30, 32
  %33 = icmp ult i32 %31, 32
  %34 = getelementptr inbounds [1 x i32], ptr %0, i32 0, i32 0
  %35 = load i32, ptr %34, align 4, !invariant.load !3
  %36 = getelementptr inbounds [1 x i32], ptr %4, i32 0, i32 0
  %37 = load i32, ptr %36, align 4, !invariant.load !3
  %38 = add i32 %37, 1
  %39 = mul nsw i64 %6, 1024
  br label %40

40:                                               ; preds = %87, %13
  %41 = phi i64 [ %88, %87 ], [ 0, %13 ]
  %42 = icmp slt i64 %41, 32
  br i1 %42, label %43, label %89

43:                                               ; preds = %40
  %44 = mul nsw i64 %41, 32
  %45 = add nsw i64 %39, %44
  br label %46

46:                                               ; preds = %49, %43
  %47 = phi i64 [ %86, %49 ], [ 0, %43 ]
  %48 = icmp slt i64 %47, 32
  br i1 %48, label %49, label %87

49:                                               ; preds = %46
  %50 = add nsw i64 %45, %47
  %51 = getelementptr inbounds [2048 x i32], ptr %2, i32 0, i64 %50
  %52 = load i32, ptr %51, align 4, !invariant.load !3
  %53 = shl i32 %52, %15
  %54 = select i1 %17, i32 %53, i32 0
  %55 = lshr i32 %52, %16
  %56 = select i1 %18, i32 %55, i32 0
  %57 = getelementptr inbounds [2048 x i32], ptr %3, i32 0, i64 %50
  %58 = load i32, ptr %57, align 4, !invariant.load !3
  %59 = or i32 %54, %56
  %60 = add i32 %58, %52
  %61 = xor i32 %60, %59
  %62 = shl i32 %61, %20
  %63 = select i1 %22, i32 %62, i32 0
  %64 = lshr i32 %61, %21
  %65 = select i1 %23, i32 %64, i32 0
  %66 = or i32 %63, %65
  %67 = add i32 %60, %61
  %68 = xor i32 %67, %66
  %69 = shl i32 %68, %25
  %70 = select i1 %27, i32 %69, i32 0
  %71 = lshr i32 %68, %26
  %72 = select i1 %28, i32 %71, i32 0
  %73 = or i32 %70, %72
  %74 = add i32 %67, %68
  %75 = xor i32 %74, %73
  %76 = shl i32 %75, %30
  %77 = select i1 %32, i32 %76, i32 0
  %78 = lshr i32 %75, %31
  %79 = select i1 %33, i32 %78, i32 0
  %80 = add i32 %74, %75
  %81 = or i32 %77, %79
  %82 = xor i32 %80, %81
  %83 = add i32 %82, %35
  %84 = add i32 %83, %38
  %85 = getelementptr inbounds [2048 x i32], ptr %5, i32 0, i64 %50
  store i32 %84, ptr %85, align 4
  %86 = add i64 %47, 1
  br label %46

87:                                               ; preds = %46
  %88 = add i64 %41, 1
  br label %40, !llvm.loop !7

89:                                               ; preds = %40
  br label %90

90:                                               ; preds = %89, %9
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 0}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 4}
!5 = !{i64 16}
!6 = !{i64 8192}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.disable"}
