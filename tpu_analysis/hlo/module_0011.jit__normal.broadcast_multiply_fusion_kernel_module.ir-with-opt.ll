; ModuleID = '__compute_module_broadcast_multiply_fusion_kernel_module'
source_filename = "__compute_module_broadcast_multiply_fusion_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define noalias noundef ptr @broadcast_multiply_fusion(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %3 = load ptr, ptr %2, align 8, !invariant.load !3
  %4 = load ptr, ptr %3, align 8, !invariant.load !3, !dereferenceable !4
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %6 = load ptr, ptr %5, align 8, !invariant.load !3, !dereferenceable !4
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %8 = load ptr, ptr %7, align 8, !invariant.load !3, !dereferenceable !4
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load ptr, ptr %9, align 8
  %11 = load i64, ptr %10, align 4, !invariant.load !3
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !8)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  %12 = icmp slt i64 %11, 5
  br i1 %12, label %13, label %120

13:                                               ; preds = %1
  %14 = icmp sgt i64 %11, -1
  br i1 %14, label %15, label %broadcast_multiply_fusion_wrapped.exit

15:                                               ; preds = %13
  %16 = mul nuw nsw i64 %11, 704
  br label %vector.ph16

vector.ph16:                                      ; preds = %15, %middle.block22
  %17 = phi i64 [ 0, %15 ], [ %119, %middle.block22 ]
  %18 = shl nuw nsw i64 %17, 6
  %19 = add nuw nsw i64 %18, %16
  br label %vector.body17

vector.body17:                                    ; preds = %vector.body17, %vector.ph16
  %index18 = phi i64 [ 0, %vector.ph16 ], [ %index.next21, %vector.body17 ]
  %20 = add nuw nsw i64 %index18, %19
  %21 = getelementptr inbounds nuw i32, ptr %4, i64 %20
  %wide.load19 = load <8 x i32>, ptr %21, align 4, !invariant.load !3, !alias.scope !5, !noalias !12
  %22 = getelementptr inbounds nuw i32, ptr %6, i64 %20
  %wide.load20 = load <8 x i32>, ptr %22, align 4, !invariant.load !3, !alias.scope !8, !noalias !13
  %23 = xor <8 x i32> %wide.load20, %wide.load19
  %24 = lshr <8 x i32> %23, splat (i32 9)
  %25 = or disjoint <8 x i32> %24, splat (i32 1065353216)
  %26 = bitcast <8 x i32> %25 to <8 x float>
  %27 = fadd <8 x float> %26, splat (float -1.000000e+00)
  %28 = fmul <8 x float> %27, splat (float 2.000000e+00)
  %29 = fadd <8 x float> %28, splat (float 0xBFEFFFFFE0000000)
  %30 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %29, <8 x float> splat (float 0xBFEFFFFFE0000000))
  %31 = fneg <8 x float> %30
  %32 = fmul <8 x float> %30, %31
  %33 = fadd <8 x float> %32, splat (float 1.000000e+00)
  %log_f32.i.i24 = fcmp ule <8 x float> %33, zeroinitializer
  %log_f323.i.i27 = fcmp une <8 x float> %33, zeroinitializer
  %log_f326.i.i30 = fcmp une <8 x float> %33, splat (float 0x7FF0000000000000)
  %.inv84 = fcmp ogt <8 x float> %33, splat (float 0x3810000000000000)
  %34 = select <8 x i1> %.inv84, <8 x float> %33, <8 x float> splat (float 0x3810000000000000)
  %35 = bitcast <8 x float> %34 to <8 x i32>
  %36 = lshr <8 x i32> %35, splat (i32 23)
  %log_f3210.i.i34 = and <8 x i32> %35, splat (i32 8388607)
  %log_f3212.i.i35 = or disjoint <8 x i32> %log_f3210.i.i34, splat (i32 1056964608)
  %log_f3213.i.i36 = bitcast <8 x i32> %log_f3212.i.i35 to <8 x float>
  %37 = add nsw <8 x i32> %36, splat (i32 -127)
  %38 = sitofp <8 x i32> %37 to <8 x float>
  %log_f3214.i.i37 = fadd <8 x float> %38, splat (float 1.000000e+00)
  %log_f3215.i.i38 = fcmp olt <8 x float> %log_f3213.i.i36, splat (float 0x3FE6A09E60000000)
  %39 = select <8 x i1> %log_f3215.i.i38, <8 x float> %log_f3213.i.i36, <8 x float> zeroinitializer
  %40 = fadd <8 x float> %log_f3213.i.i36, splat (float -1.000000e+00)
  %41 = select <8 x i1> %log_f3215.i.i38, <8 x float> splat (float 1.000000e+00), <8 x float> zeroinitializer
  %42 = fsub <8 x float> %log_f3214.i.i37, %41
  %log_f3223.i.i43 = fadd <8 x float> %40, %39
  %log_f3224.i.i44 = fmul <8 x float> %log_f3223.i.i43, %log_f3223.i.i43
  %log_f3225.i.i45 = fmul <8 x float> %log_f3224.i.i44, %log_f3223.i.i43
  %log_f3226.i.i46 = fmul <8 x float> %log_f3223.i.i43, splat (float 0x3FB2043760000000)
  %log_f3227.i.i47 = fadd <8 x float> %log_f3226.i.i46, splat (float 0xBFBD7A3700000000)
  %log_f3228.i.i48 = fmul <8 x float> %log_f3223.i.i43, splat (float 0xBFBFCBA9E0000000)
  %log_f3229.i.i49 = fadd <8 x float> %log_f3228.i.i48, splat (float 0x3FC23D37E0000000)
  %log_f3230.i.i50 = fmul <8 x float> %log_f3223.i.i43, splat (float 0x3FC999D580000000)
  %log_f3231.i.i51 = fadd <8 x float> %log_f3230.i.i50, splat (float 0xBFCFFFFF80000000)
  %log_f3232.i.i52 = fmul <8 x float> %log_f3227.i.i47, %log_f3223.i.i43
  %log_f3233.i.i53 = fadd <8 x float> %log_f3232.i.i52, splat (float 0x3FBDE4A340000000)
  %log_f3234.i.i54 = fmul <8 x float> %log_f3229.i.i49, %log_f3223.i.i43
  %log_f3235.i.i55 = fadd <8 x float> %log_f3234.i.i54, splat (float 0xBFC555CA00000000)
  %log_f3236.i.i56 = fmul <8 x float> %log_f3231.i.i51, %log_f3223.i.i43
  %log_f3237.i.i57 = fadd <8 x float> %log_f3236.i.i56, splat (float 0x3FD5555540000000)
  %log_f3238.i.i58 = fmul <8 x float> %log_f3233.i.i53, %log_f3225.i.i45
  %log_f3239.i.i59 = fadd <8 x float> %log_f3235.i.i55, %log_f3238.i.i58
  %log_f3240.i.i60 = fmul <8 x float> %log_f3239.i.i59, %log_f3225.i.i45
  %log_f3241.i.i61 = fadd <8 x float> %log_f3237.i.i57, %log_f3240.i.i60
  %log_f3242.i.i62 = fmul <8 x float> %log_f3241.i.i61, %log_f3225.i.i45
  %log_f3243.i.i63 = fmul <8 x float> %42, splat (float 0xBF2BD01060000000)
  %log_f3244.i.i64 = fmul <8 x float> %log_f3224.i.i44, splat (float 5.000000e-01)
  %log_f3245.i.i65 = fadd <8 x float> %log_f3242.i.i62, %log_f3243.i.i63
  %43 = fsub <8 x float> %log_f3223.i.i43, %log_f3244.i.i64
  %log_f3246.i.i66 = fmul <8 x float> %42, splat (float 0x3FE6300000000000)
  %log_f3247.i.i67 = fadd <8 x float> %43, %log_f3245.i.i65
  %log_f3248.i.i68 = fadd <8 x float> %log_f3247.i.i67, %log_f3246.i.i66
  %log_f3252.i.i70 = select <8 x i1> %log_f326.i.i30, <8 x i32> zeroinitializer, <8 x i32> splat (i32 2139095040)
  %log_f3255.i.i71 = select <8 x i1> %log_f323.i.i27, <8 x i32> %log_f3252.i.i70, <8 x i32> splat (i32 -8388608)
  %log_f3257.i.i73 = bitcast <8 x float> %log_f3248.i.i68 to <8 x i32>
  %log_f3259.i.i74 = select <8 x i1> %log_f32.i.i24, <8 x i32> splat (i32 -1), <8 x i32> %log_f3257.i.i73
  %log_f3263.i.i7686.not = and <8 x i1> %log_f323.i.i27, %log_f326.i.i30
  %log_f3269.i.i79 = select <8 x i1> %log_f3263.i.i7686.not, <8 x i32> %log_f3259.i.i74, <8 x i32> zeroinitializer
  %log_f3272.i.i80 = or <8 x i32> %log_f3255.i.i71, %log_f3269.i.i79
  %log_f3273.i.i81 = bitcast <8 x i32> %log_f3272.i.i80 to <8 x float>
  %44 = fmul <8 x float> %32, %32
  %45 = fmul <8 x float> %32, zeroinitializer
  %46 = fadd <8 x float> %45, splat (float 1.000000e+00)
  %47 = fmul <8 x float> %46, %32
  %48 = fadd <8 x float> %47, splat (float 0x402E2035A0000000)
  %49 = fmul <8 x float> %48, %32
  %50 = fadd <8 x float> %49, splat (float 0x4054C30B60000000)
  %51 = fmul <8 x float> %50, %32
  %52 = fadd <8 x float> %51, splat (float 0x406BB865A0000000)
  %53 = fmul <8 x float> %52, %32
  %54 = fadd <8 x float> %53, splat (float 0x4073519460000000)
  %55 = fmul <8 x float> %54, %32
  %56 = fadd <8 x float> %55, splat (float 0x406B0DB140000000)
  %57 = fmul <8 x float> %56, %32
  %58 = fadd <8 x float> %57, splat (float 0x404E0F3040000000)
  %59 = fadd <8 x float> %45, splat (float 0x3F07BC0960000000)
  %60 = fmul <8 x float> %59, %32
  %61 = fadd <8 x float> %60, splat (float 0x3FDFE818A0000000)
  %62 = fmul <8 x float> %61, %32
  %63 = fadd <8 x float> %62, splat (float 0x401A509F40000000)
  %64 = fmul <8 x float> %63, %32
  %65 = fadd <8 x float> %64, splat (float 0x403DE97380000000)
  %66 = fmul <8 x float> %65, %32
  %67 = fadd <8 x float> %66, splat (float 0x404E798EC0000000)
  %68 = fmul <8 x float> %67, %32
  %69 = fadd <8 x float> %68, splat (float 0x404C8E75A0000000)
  %70 = fmul <8 x float> %69, %32
  %71 = fadd <8 x float> %70, splat (float 0x40340A2020000000)
  %72 = fdiv <8 x float> %71, %58
  %73 = fmul <8 x float> %32, %44
  %74 = fmul <8 x float> %73, %72
  %75 = fmul <8 x float> %44, splat (float -5.000000e-01)
  %76 = fadd <8 x float> %75, %74
  %77 = fadd <8 x float> %32, %76
  %78 = call <8 x float> @llvm.fabs.v8f32(<8 x float> %32)
  %79 = fcmp olt <8 x float> %78, splat (float 0x3FDA8279A0000000)
  %80 = select <8 x i1> %79, <8 x float> %77, <8 x float> %log_f3273.i.i81
  %81 = fneg <8 x float> %80
  %82 = fcmp ogt <8 x float> %80, splat (float -5.000000e+00)
  %83 = select <8 x i1> %82, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %84 = select <8 x i1> %82, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %85 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %81)
  %86 = fsub <8 x float> splat (float -2.500000e+00), %80
  %87 = fadd <8 x float> %85, splat (float -3.000000e+00)
  %88 = select <8 x i1> %82, <8 x float> %86, <8 x float> %87
  %89 = fmul <8 x float> %83, %88
  %90 = fadd <8 x float> %84, %89
  %91 = select <8 x i1> %82, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %92 = fmul <8 x float> %88, %90
  %93 = fadd <8 x float> %91, %92
  %94 = select <8 x i1> %82, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %95 = fmul <8 x float> %88, %93
  %96 = fadd <8 x float> %94, %95
  %97 = select <8 x i1> %82, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %98 = fmul <8 x float> %88, %96
  %99 = fadd <8 x float> %97, %98
  %100 = select <8 x i1> %82, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %101 = fmul <8 x float> %88, %99
  %102 = fadd <8 x float> %100, %101
  %103 = select <8 x i1> %82, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %104 = fmul <8 x float> %88, %102
  %105 = fadd <8 x float> %103, %104
  %106 = select <8 x i1> %82, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %107 = fmul <8 x float> %88, %105
  %108 = fadd <8 x float> %106, %107
  %109 = select <8 x i1> %82, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %110 = fmul <8 x float> %88, %108
  %111 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %30)
  %112 = fadd <8 x float> %109, %110
  %113 = fcmp oeq <8 x float> %111, splat (float 1.000000e+00)
  %114 = select <8 x i1> %113, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %112
  %115 = fmul <8 x float> %30, %114
  %116 = fmul <8 x float> %115, splat (float 0x3FF6A09E60000000)
  %117 = getelementptr inbounds nuw float, ptr %8, i64 %20
  store <8 x float> %116, ptr %117, align 4, !alias.scope !10, !noalias !14
  %index.next21 = add nuw i64 %index18, 8
  %118 = icmp eq i64 %index.next21, 64
  br i1 %118, label %middle.block22, label %vector.body17, !llvm.loop !15

middle.block22:                                   ; preds = %vector.body17
  %119 = add nuw nsw i64 %17, 1
  %exitcond10.not = icmp eq i64 %119, 11
  br i1 %exitcond10.not, label %broadcast_multiply_fusion_wrapped.exit, label %vector.ph16, !llvm.loop !18

120:                                              ; preds = %1
  %121 = icmp eq i64 %11, 5
  br i1 %121, label %.preheader, label %broadcast_multiply_fusion_wrapped.exit

.preheader:                                       ; preds = %120, %middle.block
  %122 = phi i64 [ %224, %middle.block ], [ 0, %120 ]
  %123 = shl nuw nsw i64 %122, 6
  %124 = add nuw nsw i64 %123, 3520
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %.preheader
  %index = phi i64 [ 0, %.preheader ], [ %index.next, %vector.body ]
  %125 = add nuw nsw i64 %124, %index
  %126 = getelementptr inbounds nuw i32, ptr %4, i64 %125
  %wide.load = load <8 x i32>, ptr %126, align 4, !invariant.load !3, !alias.scope !5, !noalias !12
  %127 = getelementptr inbounds nuw i32, ptr %6, i64 %125
  %wide.load15 = load <8 x i32>, ptr %127, align 4, !invariant.load !3, !alias.scope !8, !noalias !13
  %128 = xor <8 x i32> %wide.load15, %wide.load
  %129 = lshr <8 x i32> %128, splat (i32 9)
  %130 = or disjoint <8 x i32> %129, splat (i32 1065353216)
  %131 = bitcast <8 x i32> %130 to <8 x float>
  %132 = fadd <8 x float> %131, splat (float -1.000000e+00)
  %133 = fmul <8 x float> %132, splat (float 2.000000e+00)
  %134 = fadd <8 x float> %133, splat (float 0xBFEFFFFFE0000000)
  %135 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %134, <8 x float> splat (float 0xBFEFFFFFE0000000))
  %136 = fneg <8 x float> %135
  %137 = fmul <8 x float> %135, %136
  %138 = fadd <8 x float> %137, splat (float 1.000000e+00)
  %log_f32.i.i = fcmp ule <8 x float> %138, zeroinitializer
  %log_f323.i.i = fcmp une <8 x float> %138, zeroinitializer
  %log_f326.i.i = fcmp une <8 x float> %138, splat (float 0x7FF0000000000000)
  %.inv = fcmp ogt <8 x float> %138, splat (float 0x3810000000000000)
  %139 = select <8 x i1> %.inv, <8 x float> %138, <8 x float> splat (float 0x3810000000000000)
  %140 = bitcast <8 x float> %139 to <8 x i32>
  %141 = lshr <8 x i32> %140, splat (i32 23)
  %log_f3210.i.i = and <8 x i32> %140, splat (i32 8388607)
  %log_f3212.i.i = or disjoint <8 x i32> %log_f3210.i.i, splat (i32 1056964608)
  %log_f3213.i.i = bitcast <8 x i32> %log_f3212.i.i to <8 x float>
  %142 = add nsw <8 x i32> %141, splat (i32 -127)
  %143 = sitofp <8 x i32> %142 to <8 x float>
  %log_f3214.i.i = fadd <8 x float> %143, splat (float 1.000000e+00)
  %log_f3215.i.i = fcmp olt <8 x float> %log_f3213.i.i, splat (float 0x3FE6A09E60000000)
  %144 = select <8 x i1> %log_f3215.i.i, <8 x float> %log_f3213.i.i, <8 x float> zeroinitializer
  %145 = fadd <8 x float> %log_f3213.i.i, splat (float -1.000000e+00)
  %146 = select <8 x i1> %log_f3215.i.i, <8 x float> splat (float 1.000000e+00), <8 x float> zeroinitializer
  %147 = fsub <8 x float> %log_f3214.i.i, %146
  %log_f3223.i.i = fadd <8 x float> %145, %144
  %log_f3224.i.i = fmul <8 x float> %log_f3223.i.i, %log_f3223.i.i
  %log_f3225.i.i = fmul <8 x float> %log_f3224.i.i, %log_f3223.i.i
  %log_f3226.i.i = fmul <8 x float> %log_f3223.i.i, splat (float 0x3FB2043760000000)
  %log_f3227.i.i = fadd <8 x float> %log_f3226.i.i, splat (float 0xBFBD7A3700000000)
  %log_f3228.i.i = fmul <8 x float> %log_f3223.i.i, splat (float 0xBFBFCBA9E0000000)
  %log_f3229.i.i = fadd <8 x float> %log_f3228.i.i, splat (float 0x3FC23D37E0000000)
  %log_f3230.i.i = fmul <8 x float> %log_f3223.i.i, splat (float 0x3FC999D580000000)
  %log_f3231.i.i = fadd <8 x float> %log_f3230.i.i, splat (float 0xBFCFFFFF80000000)
  %log_f3232.i.i = fmul <8 x float> %log_f3227.i.i, %log_f3223.i.i
  %log_f3233.i.i = fadd <8 x float> %log_f3232.i.i, splat (float 0x3FBDE4A340000000)
  %log_f3234.i.i = fmul <8 x float> %log_f3229.i.i, %log_f3223.i.i
  %log_f3235.i.i = fadd <8 x float> %log_f3234.i.i, splat (float 0xBFC555CA00000000)
  %log_f3236.i.i = fmul <8 x float> %log_f3231.i.i, %log_f3223.i.i
  %log_f3237.i.i = fadd <8 x float> %log_f3236.i.i, splat (float 0x3FD5555540000000)
  %log_f3238.i.i = fmul <8 x float> %log_f3233.i.i, %log_f3225.i.i
  %log_f3239.i.i = fadd <8 x float> %log_f3235.i.i, %log_f3238.i.i
  %log_f3240.i.i = fmul <8 x float> %log_f3239.i.i, %log_f3225.i.i
  %log_f3241.i.i = fadd <8 x float> %log_f3237.i.i, %log_f3240.i.i
  %log_f3242.i.i = fmul <8 x float> %log_f3241.i.i, %log_f3225.i.i
  %log_f3243.i.i = fmul <8 x float> %147, splat (float 0xBF2BD01060000000)
  %log_f3244.i.i = fmul <8 x float> %log_f3224.i.i, splat (float 5.000000e-01)
  %log_f3245.i.i = fadd <8 x float> %log_f3242.i.i, %log_f3243.i.i
  %148 = fsub <8 x float> %log_f3223.i.i, %log_f3244.i.i
  %log_f3246.i.i = fmul <8 x float> %147, splat (float 0x3FE6300000000000)
  %log_f3247.i.i = fadd <8 x float> %148, %log_f3245.i.i
  %log_f3248.i.i = fadd <8 x float> %log_f3247.i.i, %log_f3246.i.i
  %log_f3252.i.i = select <8 x i1> %log_f326.i.i, <8 x i32> zeroinitializer, <8 x i32> splat (i32 2139095040)
  %log_f3255.i.i = select <8 x i1> %log_f323.i.i, <8 x i32> %log_f3252.i.i, <8 x i32> splat (i32 -8388608)
  %log_f3257.i.i = bitcast <8 x float> %log_f3248.i.i to <8 x i32>
  %log_f3259.i.i = select <8 x i1> %log_f32.i.i, <8 x i32> splat (i32 -1), <8 x i32> %log_f3257.i.i
  %log_f3263.i.i83.not = and <8 x i1> %log_f323.i.i, %log_f326.i.i
  %log_f3269.i.i = select <8 x i1> %log_f3263.i.i83.not, <8 x i32> %log_f3259.i.i, <8 x i32> zeroinitializer
  %log_f3272.i.i = or <8 x i32> %log_f3255.i.i, %log_f3269.i.i
  %log_f3273.i.i = bitcast <8 x i32> %log_f3272.i.i to <8 x float>
  %149 = fmul <8 x float> %137, %137
  %150 = fmul <8 x float> %137, zeroinitializer
  %151 = fadd <8 x float> %150, splat (float 1.000000e+00)
  %152 = fmul <8 x float> %151, %137
  %153 = fadd <8 x float> %152, splat (float 0x402E2035A0000000)
  %154 = fmul <8 x float> %153, %137
  %155 = fadd <8 x float> %154, splat (float 0x4054C30B60000000)
  %156 = fmul <8 x float> %155, %137
  %157 = fadd <8 x float> %156, splat (float 0x406BB865A0000000)
  %158 = fmul <8 x float> %157, %137
  %159 = fadd <8 x float> %158, splat (float 0x4073519460000000)
  %160 = fmul <8 x float> %159, %137
  %161 = fadd <8 x float> %160, splat (float 0x406B0DB140000000)
  %162 = fmul <8 x float> %161, %137
  %163 = fadd <8 x float> %162, splat (float 0x404E0F3040000000)
  %164 = fadd <8 x float> %150, splat (float 0x3F07BC0960000000)
  %165 = fmul <8 x float> %164, %137
  %166 = fadd <8 x float> %165, splat (float 0x3FDFE818A0000000)
  %167 = fmul <8 x float> %166, %137
  %168 = fadd <8 x float> %167, splat (float 0x401A509F40000000)
  %169 = fmul <8 x float> %168, %137
  %170 = fadd <8 x float> %169, splat (float 0x403DE97380000000)
  %171 = fmul <8 x float> %170, %137
  %172 = fadd <8 x float> %171, splat (float 0x404E798EC0000000)
  %173 = fmul <8 x float> %172, %137
  %174 = fadd <8 x float> %173, splat (float 0x404C8E75A0000000)
  %175 = fmul <8 x float> %174, %137
  %176 = fadd <8 x float> %175, splat (float 0x40340A2020000000)
  %177 = fdiv <8 x float> %176, %163
  %178 = fmul <8 x float> %137, %149
  %179 = fmul <8 x float> %178, %177
  %180 = fmul <8 x float> %149, splat (float -5.000000e-01)
  %181 = fadd <8 x float> %180, %179
  %182 = fadd <8 x float> %137, %181
  %183 = call <8 x float> @llvm.fabs.v8f32(<8 x float> %137)
  %184 = fcmp olt <8 x float> %183, splat (float 0x3FDA8279A0000000)
  %185 = select <8 x i1> %184, <8 x float> %182, <8 x float> %log_f3273.i.i
  %186 = fneg <8 x float> %185
  %187 = fcmp ogt <8 x float> %185, splat (float -5.000000e+00)
  %188 = select <8 x i1> %187, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %189 = select <8 x i1> %187, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %190 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %186)
  %191 = fsub <8 x float> splat (float -2.500000e+00), %185
  %192 = fadd <8 x float> %190, splat (float -3.000000e+00)
  %193 = select <8 x i1> %187, <8 x float> %191, <8 x float> %192
  %194 = fmul <8 x float> %188, %193
  %195 = fadd <8 x float> %189, %194
  %196 = select <8 x i1> %187, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %197 = fmul <8 x float> %193, %195
  %198 = fadd <8 x float> %196, %197
  %199 = select <8 x i1> %187, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %200 = fmul <8 x float> %193, %198
  %201 = fadd <8 x float> %199, %200
  %202 = select <8 x i1> %187, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %203 = fmul <8 x float> %193, %201
  %204 = fadd <8 x float> %202, %203
  %205 = select <8 x i1> %187, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %206 = fmul <8 x float> %193, %204
  %207 = fadd <8 x float> %205, %206
  %208 = select <8 x i1> %187, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %209 = fmul <8 x float> %193, %207
  %210 = fadd <8 x float> %208, %209
  %211 = select <8 x i1> %187, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %212 = fmul <8 x float> %193, %210
  %213 = fadd <8 x float> %211, %212
  %214 = select <8 x i1> %187, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %215 = fmul <8 x float> %193, %213
  %216 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %135)
  %217 = fadd <8 x float> %214, %215
  %218 = fcmp oeq <8 x float> %216, splat (float 1.000000e+00)
  %219 = select <8 x i1> %218, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %217
  %220 = fmul <8 x float> %135, %219
  %221 = fmul <8 x float> %220, splat (float 0x3FF6A09E60000000)
  %222 = getelementptr inbounds nuw float, ptr %8, i64 %125
  store <8 x float> %221, ptr %222, align 4, !alias.scope !10, !noalias !14
  %index.next = add nuw i64 %index, 8
  %223 = icmp eq i64 %index.next, 64
  br i1 %223, label %middle.block, label %vector.body, !llvm.loop !20

middle.block:                                     ; preds = %vector.body
  %224 = add nuw nsw i64 %122, 1
  %exitcond8.not = icmp eq i64 %224, 9
  br i1 %exitcond8.not, label %broadcast_multiply_fusion_wrapped.exit, label %.preheader, !llvm.loop !18

broadcast_multiply_fusion_wrapped.exit:           ; preds = %middle.block, %middle.block22, %13, %120
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!xla_cpu_memory_region_name = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"xla_dylib_index", i64 4}
!2 = !{!"xla_cpu_emitter__loop_fusion_kernel_emitter__hlo_opcode__fusion"}
!3 = !{}
!4 = !{i64 16384}
!5 = !{!6}
!6 = distinct !{!6, !7, !"broadcast_multiply_fusion_wrapped: argument 0"}
!7 = distinct !{!7, !"broadcast_multiply_fusion_wrapped"}
!8 = !{!9}
!9 = distinct !{!9, !7, !"broadcast_multiply_fusion_wrapped: argument 1"}
!10 = !{!11}
!11 = distinct !{!11, !7, !"broadcast_multiply_fusion_wrapped: argument 2"}
!12 = !{!9, !11}
!13 = !{!6, !11}
!14 = !{!6, !9}
!15 = distinct !{!15, !16, !17}
!16 = !{!"llvm.loop.isvectorized", i32 1}
!17 = !{!"llvm.loop.unroll.runtime.disable"}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.unroll.disable"}
!20 = distinct !{!20, !16, !17}
