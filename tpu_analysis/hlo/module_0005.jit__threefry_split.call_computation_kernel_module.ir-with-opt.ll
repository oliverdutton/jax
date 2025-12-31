; ModuleID = '__compute_module_call_computation_kernel_module'
source_filename = "__compute_module_call_computation_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) uwtable
define noalias noundef ptr @call_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg20_gep = getelementptr i8, ptr %args, i64 320
  %arg20 = load ptr, ptr %arg20_gep, align 8, !invariant.load !3, !dereferenceable !4, !align !5
  %arg21_gep = getelementptr i8, ptr %args, i64 336
  %arg21 = load ptr, ptr %arg21_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg22_gep = getelementptr i8, ptr %args, i64 352
  %arg22 = load ptr, ptr %arg22_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg23_gep = getelementptr i8, ptr %args, i64 368
  %arg23 = load ptr, ptr %arg23_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg24_gep = getelementptr i8, ptr %args, i64 384
  %arg24 = load ptr, ptr %arg24_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg25_gep = getelementptr i8, ptr %args, i64 400
  %arg25 = load ptr, ptr %arg25_gep, align 8, !invariant.load !3, !dereferenceable !8, !align !5
  %arg26_gep = getelementptr i8, ptr %args, i64 416
  %arg26 = load ptr, ptr %arg26_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg27_gep = getelementptr i8, ptr %args, i64 432
  %arg27 = load ptr, ptr %arg27_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg28_gep = getelementptr i8, ptr %args, i64 448
  %arg28 = load ptr, ptr %arg28_gep, align 8, !invariant.load !3, !dereferenceable !8, !align !5
  %arg29_gep = getelementptr i8, ptr %args, i64 464
  %arg29 = load ptr, ptr %arg29_gep, align 8, !invariant.load !3, !dereferenceable !8, !align !5
  %arg30_gep = getelementptr i8, ptr %args, i64 480
  %arg30 = load ptr, ptr %arg30_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg31_gep = getelementptr i8, ptr %args, i64 496
  %arg31 = load ptr, ptr %arg31_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg32_gep = getelementptr i8, ptr %args, i64 512
  %arg32 = load ptr, ptr %arg32_gep, align 8, !invariant.load !3, !dereferenceable !5, !align !5
  %arg34_gep = getelementptr i8, ptr %args, i64 544
  %arg34 = load ptr, ptr %arg34_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg35_gep = getelementptr i8, ptr %args, i64 560
  %arg35 = load ptr, ptr %arg35_gep, align 8, !invariant.load !3, !dereferenceable !8, !align !5
  %arg36_gep = getelementptr i8, ptr %args, i64 576
  %arg36 = load ptr, ptr %arg36_gep, align 8, !invariant.load !3, !dereferenceable !6, !align !5
  %arg37_gep = getelementptr i8, ptr %args, i64 592
  %arg37 = load ptr, ptr %arg37_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %arg38_gep = getelementptr i8, ptr %args, i64 608
  %arg38 = load ptr, ptr %arg38_gep, align 8, !invariant.load !3, !dereferenceable !7, !align !5
  %2 = load i32, ptr %arg31, align 64, !alias.scope !9, !noalias !12
  %3 = icmp slt i32 %2, 5
  %4 = zext i1 %3 to i8
  store i8 %4, ptr %arg20, align 64, !alias.scope !19, !noalias !20
  br i1 %3, label %while.6.body.i.lr.ph, label %return

while.6.body.i.lr.ph:                             ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %arg25, i64 4
  %6 = getelementptr inbounds nuw i8, ptr %arg25, i64 8
  %7 = getelementptr inbounds nuw i8, ptr %arg25, i64 12
  %8 = getelementptr inbounds nuw i8, ptr %arg32, i64 8
  %9 = getelementptr inbounds nuw i8, ptr %arg32, i64 16
  %10 = getelementptr inbounds nuw i8, ptr %arg32, i64 24
  %11 = getelementptr inbounds nuw i8, ptr %arg32, i64 32
  %12 = getelementptr inbounds nuw i8, ptr %arg32, i64 40
  %13 = getelementptr inbounds nuw i8, ptr %arg32, i64 48
  %14 = getelementptr inbounds nuw i8, ptr %arg32, i64 56
  %15 = getelementptr inbounds nuw i8, ptr %arg26, i64 4
  %16 = getelementptr inbounds nuw i8, ptr %arg36, i64 4
  %17 = getelementptr inbounds nuw i8, ptr %arg21, i64 4
  %18 = getelementptr inbounds nuw i8, ptr %arg26, i64 8
  %19 = getelementptr inbounds nuw i8, ptr %arg36, i64 8
  %20 = getelementptr inbounds nuw i8, ptr %arg21, i64 8
  %21 = getelementptr inbounds nuw i8, ptr %arg24, i64 4
  %22 = getelementptr inbounds nuw i8, ptr %arg24, i64 8
  br label %while.6.body.i

while.6.body.i:                                   ; preds = %while.6.body.i.lr.ph, %while.6.body.i
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg28, ptr noundef nonnull align 64 dereferenceable(16) %arg35, i64 16, i1 false), !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg25, ptr noundef nonnull align 64 dereferenceable(16) %arg29, i64 16, i1 false), !noalias !21
  %23 = load i32, ptr %arg23, align 64, !noalias !21
  store i32 %23, ptr %arg27, align 64, !noalias !21
  %24 = load i32, ptr %arg30, align 64, !noalias !21
  store i32 %24, ptr %arg38, align 64, !noalias !21
  %25 = load i32, ptr %arg22, align 64, !noalias !21
  store i32 %25, ptr %arg34, align 64, !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(12) %arg36, ptr noundef nonnull align 64 dereferenceable(12) %arg24, i64 12, i1 false), !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(12) %arg26, ptr noundef nonnull align 64 dereferenceable(12) %arg21, i64 12, i1 false), !noalias !21
  %26 = load i32, ptr %arg31, align 64, !noalias !21
  store i32 %26, ptr %arg37, align 64, !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg35, ptr noundef nonnull align 64 dereferenceable(16) %arg25, i64 16, i1 false), !noalias !21
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg29, ptr noundef nonnull align 64 dereferenceable(16) %arg28, i64 16, i1 false), !noalias !21
  %27 = load i32, ptr %arg27, align 64, !noalias !21
  store i32 %27, ptr %arg30, align 64, !noalias !21
  %28 = load i32, ptr %arg34, align 64, !noalias !21
  store i32 %28, ptr %arg23, align 64, !noalias !21
  %29 = load i32, ptr %arg38, align 64, !noalias !21
  store i32 %29, ptr %arg22, align 64, !noalias !21
  %30 = load i32, ptr %arg25, align 64, !alias.scope !24, !noalias !26
  %shft.chk.i.i = icmp ult i32 %30, 32
  %31 = sub i32 32, %30
  %shft.chk1.i.i = icmp ult i32 %31, 32
  %32 = load i32, ptr %5, align 4, !alias.scope !24, !noalias !26
  %shft.chk2.i.i = icmp ult i32 %32, 32
  %33 = sub i32 32, %32
  %shft.chk4.i.i = icmp ult i32 %33, 32
  %34 = load i32, ptr %6, align 8, !alias.scope !24, !noalias !26
  %shft.chk5.i.i = icmp ult i32 %34, 32
  %35 = sub i32 32, %34
  %shft.chk7.i.i = icmp ult i32 %35, 32
  %36 = load i32, ptr %arg34, align 64, !alias.scope !36, !noalias !37
  %37 = load i32, ptr %arg26, align 64, !alias.scope !40, !noalias !41
  %38 = load i32, ptr %arg36, align 64, !alias.scope !42, !noalias !43
  %39 = add i32 %38, %37
  %40 = shl i32 %38, %30
  %41 = select i1 %shft.chk.i.i, i32 %40, i32 0
  %42 = lshr i32 %38, %31
  %43 = select i1 %shft.chk1.i.i, i32 %42, i32 0
  %44 = or i32 %43, %41
  %45 = xor i32 %44, %39
  %46 = add i32 %45, %39
  %47 = shl i32 %45, %32
  %48 = select i1 %shft.chk2.i.i, i32 %47, i32 0
  %49 = lshr i32 %45, %33
  %50 = select i1 %shft.chk4.i.i, i32 %49, i32 0
  %51 = or i32 %48, %50
  %52 = xor i32 %51, %46
  %53 = add i32 %52, %46
  %54 = shl i32 %52, %34
  %55 = select i1 %shft.chk5.i.i, i32 %54, i32 0
  %56 = lshr i32 %52, %35
  %57 = select i1 %shft.chk7.i.i, i32 %56, i32 0
  %58 = or i32 %55, %57
  %59 = xor i32 %58, %53
  %60 = add i32 %53, %36
  %61 = add i32 %60, %59
  store i32 %61, ptr %arg21, align 64, !alias.scope !44, !noalias !45
  %62 = load i32, ptr %15, align 4, !alias.scope !40, !noalias !41
  %63 = load i32, ptr %16, align 4, !alias.scope !42, !noalias !43
  %64 = add i32 %63, %62
  %65 = shl i32 %63, %30
  %66 = select i1 %shft.chk.i.i, i32 %65, i32 0
  %67 = lshr i32 %63, %31
  %68 = select i1 %shft.chk1.i.i, i32 %67, i32 0
  %69 = or i32 %68, %66
  %70 = xor i32 %69, %64
  %71 = add i32 %70, %64
  %72 = shl i32 %70, %32
  %73 = select i1 %shft.chk2.i.i, i32 %72, i32 0
  %74 = lshr i32 %70, %33
  %75 = select i1 %shft.chk4.i.i, i32 %74, i32 0
  %76 = or i32 %73, %75
  %77 = xor i32 %76, %71
  %78 = add i32 %77, %71
  %79 = shl i32 %77, %34
  %80 = select i1 %shft.chk5.i.i, i32 %79, i32 0
  %81 = lshr i32 %77, %35
  %82 = select i1 %shft.chk7.i.i, i32 %81, i32 0
  %83 = or i32 %80, %82
  %84 = xor i32 %83, %78
  %85 = add i32 %78, %36
  %86 = add i32 %85, %84
  store i32 %86, ptr %17, align 4, !alias.scope !44, !noalias !45
  %87 = load i32, ptr %18, align 8, !alias.scope !40, !noalias !41
  %88 = load i32, ptr %19, align 8, !alias.scope !42, !noalias !43
  %89 = add i32 %88, %87
  %90 = shl i32 %88, %30
  %91 = select i1 %shft.chk.i.i, i32 %90, i32 0
  %92 = lshr i32 %88, %31
  %93 = select i1 %shft.chk1.i.i, i32 %92, i32 0
  %94 = or i32 %93, %91
  %95 = xor i32 %94, %89
  %96 = add i32 %95, %89
  %97 = shl i32 %95, %32
  %98 = select i1 %shft.chk2.i.i, i32 %97, i32 0
  %99 = lshr i32 %95, %33
  %100 = select i1 %shft.chk4.i.i, i32 %99, i32 0
  %101 = or i32 %98, %100
  %102 = xor i32 %101, %96
  %103 = add i32 %102, %96
  %104 = shl i32 %102, %34
  %105 = select i1 %shft.chk5.i.i, i32 %104, i32 0
  %106 = lshr i32 %102, %35
  %107 = select i1 %shft.chk7.i.i, i32 %106, i32 0
  %108 = or i32 %105, %107
  %109 = xor i32 %108, %103
  %110 = add i32 %103, %36
  %111 = add i32 %110, %109
  store i32 %111, ptr %20, align 8, !alias.scope !44, !noalias !45
  %112 = load i32, ptr %7, align 4, !alias.scope !24, !noalias !26
  %shft.chk17.i.i = icmp ult i32 %112, 32
  %113 = sub i32 32, %112
  %shft.chk19.i.i = icmp ult i32 %113, 32
  %114 = load i32, ptr %arg38, align 64, !alias.scope !48, !noalias !49
  %115 = load i32, ptr %arg37, align 64, !alias.scope !50, !noalias !51
  %116 = add i32 %114, 1
  %117 = add i32 %116, %115
  %118 = add i32 %59, %53
  %119 = shl i32 %59, %112
  %120 = select i1 %shft.chk17.i.i, i32 %119, i32 0
  %121 = lshr i32 %59, %113
  %122 = select i1 %shft.chk19.i.i, i32 %121, i32 0
  %123 = or i32 %120, %122
  %124 = xor i32 %123, %118
  %125 = add i32 %117, %124
  store i32 %125, ptr %arg24, align 64, !alias.scope !53, !noalias !54
  %126 = add i32 %84, %78
  %127 = shl i32 %84, %112
  %128 = select i1 %shft.chk17.i.i, i32 %127, i32 0
  %129 = lshr i32 %84, %113
  %130 = select i1 %shft.chk19.i.i, i32 %129, i32 0
  %131 = or i32 %128, %130
  %132 = xor i32 %131, %126
  %133 = add i32 %117, %132
  store i32 %133, ptr %21, align 4, !alias.scope !53, !noalias !54
  %134 = add i32 %109, %103
  %135 = shl i32 %109, %112
  %136 = select i1 %shft.chk17.i.i, i32 %135, i32 0
  %137 = lshr i32 %109, %113
  %138 = select i1 %shft.chk19.i.i, i32 %137, i32 0
  %139 = or i32 %136, %138
  %140 = xor i32 %139, %134
  %141 = add i32 %117, %140
  store i32 %141, ptr %22, align 8, !alias.scope !53, !noalias !54
  %142 = add i32 %115, 1
  store i32 %142, ptr %arg31, align 64, !alias.scope !9, !noalias !55
  store ptr %arg31, ptr %arg32, align 64, !alias.scope !56, !noalias !57
  store ptr %arg21, ptr %8, align 8, !alias.scope !56, !noalias !57
  store ptr %arg24, ptr %9, align 16, !alias.scope !56, !noalias !57
  store ptr %arg22, ptr %10, align 8, !alias.scope !56, !noalias !57
  store ptr %arg30, ptr %11, align 32, !alias.scope !56, !noalias !57
  store ptr %arg23, ptr %12, align 8, !alias.scope !56, !noalias !57
  store ptr %arg29, ptr %13, align 16, !alias.scope !56, !noalias !57
  store ptr %arg35, ptr %14, align 8, !alias.scope !56, !noalias !57
  %143 = icmp slt i32 %142, 5
  %144 = zext i1 %143 to i8
  store i8 %144, ptr %arg20, align 64, !alias.scope !19, !noalias !20
  br i1 %143, label %while.6.body.i, label %return

return:                                           ; preds = %while.6.body.i, %1
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!xla_cpu_memory_region_name = !{!0, !1}
!llvm.module.flags = !{!2}

!0 = !{!"xla_cpu_emitter__computation_kernel_emitter__hlo_opcode__call"}
!1 = !{!"ir_emitter"}
!2 = !{i32 1, !"xla_dylib_index", i64 0}
!3 = !{}
!4 = !{i64 1}
!5 = !{i64 64}
!6 = !{i64 12}
!7 = !{i64 4}
!8 = !{i64 16}
!9 = !{!10}
!10 = !{!"buffer: {index:7, offset:768, size:4}", !11}
!11 = !{!"XLA global AA domain"}
!12 = !{!13, !14, !15, !17}
!13 = !{!"buffer: {index:6, offset:0, size:4}", !11}
!14 = !{!"buffer: {index:7, offset:64, size:1}", !11}
!15 = distinct !{!15, !16, !"while.6__1: %buffer_table"}
!16 = distinct !{!16, !"while.6__1"}
!17 = distinct !{!17, !18, !"while.5_computation: %buffer_table"}
!18 = distinct !{!18, !"while.5_computation"}
!19 = !{!14}
!20 = !{!13, !10, !15, !17}
!21 = !{!22, !17}
!22 = distinct !{!22, !23, !"while.6: %buffer_table"}
!23 = distinct !{!23, !"while.6"}
!24 = !{!25}
!25 = !{!"buffer: {index:7, offset:64, size:16}", !11}
!26 = !{!27, !28, !29, !30, !31, !32, !33, !34, !35, !22, !17}
!27 = !{!"buffer: {index:1, offset:0, size:16}", !11}
!28 = !{!"buffer: {index:7, offset:192, size:16}", !11}
!29 = !{!"buffer: {index:7, offset:256, size:12}", !11}
!30 = !{!"buffer: {index:7, offset:320, size:12}", !11}
!31 = !{!"buffer: {index:7, offset:384, size:12}", !11}
!32 = !{!"buffer: {index:7, offset:448, size:12}", !11}
!33 = !{!"buffer: {index:7, offset:512, size:4}", !11}
!34 = !{!"buffer: {index:7, offset:576, size:4}", !11}
!35 = !{!"buffer: {index:7, offset:640, size:4}", !11}
!36 = !{!35}
!37 = !{!25, !29, !30, !31, !38, !39, !22, !17}
!38 = !{!"buffer: {index:7, offset:832, size:4}", !11}
!39 = !{!"buffer: {index:7, offset:960, size:4}", !11}
!40 = !{!30}
!41 = !{!25, !29, !31, !32, !33, !34, !35, !22, !17}
!42 = !{!29}
!43 = !{!25, !30, !31, !32, !33, !34, !35, !22, !17}
!44 = !{!31}
!45 = !{!27, !46, !25, !28, !29, !30, !32, !35, !10, !38, !47, !39, !22, !17}
!46 = !{!"buffer: {index:7, offset:0, size:64}", !11}
!47 = !{!"buffer: {index:7, offset:896, size:4}", !11}
!48 = !{!33}
!49 = !{!25, !29, !30, !32, !34, !38, !47, !22, !17}
!50 = !{!34}
!51 = !{!52, !25, !29, !30, !32, !33, !10, !22, !17}
!52 = !{!"buffer: {index:0, offset:0, size:4}", !11}
!53 = !{!32}
!54 = !{!27, !46, !25, !28, !29, !30, !31, !33, !34, !10, !38, !47, !39, !22, !17}
!55 = !{!52, !27, !46, !28, !31, !32, !34, !38, !47, !39, !22, !17}
!56 = !{!46}
!57 = !{!27, !28, !31, !32, !10, !38, !47, !39, !22, !17}
