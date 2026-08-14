; ModuleID = '/home/kranex/dev/rust-metaprogramming/examples/oxide/ripple.linked.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1
@__cudart_i2opi_f = internal unnamed_addr addrspace(1) constant [6 x i32] [i32 1011060801, i32 -614296167, i32 -181084736, i32 -64530479, i32 1313084713, i32 -1560706194], align 4
@llvm.used = appending global [1 x ptr] [ptr @ripple], section "llvm.metadata"

; Function Attrs: convergent memory(argmem: write)
define ptx_kernel void @ripple(ptr writeonly captures(address_is_null) %v0, i64 %v1, i32 %v2, float %v3) #0 {
entry:
  %result.i.i.i.i = alloca [7 x i32], align 4
  %v6.i = zext i32 %v2 to i64
  %v7.i = icmp eq i32 %v2, 0
  br i1 %v7.i, label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, label %bb24.i

bb3.i:                                            ; preds = %bb24.i
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #6
  %v17.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #6
  %v18.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #6
  %narrow.i = mul nuw nsw i32 %v13.i, %v17.i
  %narrow3.i = add nuw nsw i32 %narrow.i, %v18.i
  %v68.i = zext nneg i32 %narrow3.i to i64
  %v70.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #6
  %v23.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #6
  %v24.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #6
  %v25.i = zext nneg i32 %v70.i to i64
  %v26.i = zext nneg i32 %v23.i to i64
  %v27.i = zext nneg i32 %v24.i to i64
  %v75.i = mul nuw nsw i64 %v25.i, %v26.i
  %v76.i = add nuw nsw i64 %v75.i, %v27.i
  %v30.not.i.not = icmp samesign ult i64 %v76.i, %v6.i
  %v42.i = shl nuw nsw i64 %v68.i, 32
  %v43.i = or i64 %v42.i, %v76.i
  %spec.select15 = select i1 %v30.not.i.not, i64 %v43.i, i64 undef
  br label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit

bb24.i:                                           ; preds = %entry
  %v57.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #6
  %v58.i = icmp ne i32 %v57.i, 1
  %v59.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #6
  %v60.i = icmp ne i32 %v59.i, 1
  %v61.i = or i1 %v58.i, %v60.i
  br i1 %v61.i, label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, label %bb3.i

_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit: ; preds = %bb3.i, %entry, %bb24.i
  %v51.i = phi i1 [ false, %entry ], [ false, %bb24.i ], [ %v30.not.i.not, %bb3.i ]
  %v52.i = phi i64 [ undef, %entry ], [ undef, %bb24.i ], [ %spec.select15, %bb3.i ]
  br i1 %v51.i, label %bb3, label %bb10

bb3:                                              ; preds = %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit
  %v21 = trunc i64 %v52.i to i32
  %v24 = lshr i64 %v52.i, 32
  %v25 = trunc nuw i64 %v24 to i32
  %v28 = uitofp i32 %v2 to float
  %v29 = uitofp i32 %v21 to float
  %v30 = fmul contract float %v29, 5.000000e-01
  %v31 = fdiv contract float %v28, 1.500000e+01
  %v32 = fsub contract float %v30, %v31
  %v33 = uitofp i32 %v25 to float
  %v34 = fmul contract float %v33, 5.000000e-01
  %v35 = fsub contract float %v34, %v31
  %v36 = fmul contract float %v32, %v32
  %v37 = fmul contract float %v35, %v35
  %v38 = fadd contract float %v36, %v37
  %0 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #7
  %.not.i = icmp eq i32 %0, 0
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #7
  %.not1.i = icmp eq i32 %1, 0
  br i1 %.not.i, label %7, label %2

2:                                                ; preds = %bb3
  br i1 %.not1.i, label %5, label %3

3:                                                ; preds = %2
  %4 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v38) #7
  br label %__nv_sqrtf.exit

5:                                                ; preds = %2
  %6 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v38) #7
  br label %__nv_sqrtf.exit

7:                                                ; preds = %bb3
  br i1 %.not1.i, label %10, label %8

8:                                                ; preds = %7
  %9 = tail call float @llvm.nvvm.sqrt.rn.f(float %v38) #7
  br label %__nv_sqrtf.exit

10:                                               ; preds = %7
  %11 = tail call float @llvm.nvvm.sqrt.approx.f(float %v38) #7
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %3, %5, %8, %10
  %.0.i = phi float [ %4, %3 ], [ %6, %5 ], [ %9, %8 ], [ %11, %10 ]
  %v55 = fdiv contract float %.0.i, 1.000000e+01
  %v56 = fdiv contract float %v3, 7.000000e+00
  %v57 = fsub contract float %v55, %v56
  call void @llvm.lifetime.start.p0(i64 28, ptr nonnull %result.i.i.i.i)
  %12 = fmul float %v57, 0x3FE45F3060000000
  %13 = tail call i32 @llvm.nvvm.f2i.rn.ftz(float %12) #7
  %14 = tail call i32 @llvm.nvvm.f2i.rn(float %12) #7
  %.01.i = select i1 %.not.i, i32 %14, i32 %13
  %15 = sitofp i32 %.01.i to float
  %16 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %15, float 0xBFF921FB40000000, float %v57) #7
  %17 = tail call float @llvm.fma.f32(float %15, float 0xBFF921FB40000000, float %v57)
  %.02.i = select i1 %.not.i, float %17, float %16
  %18 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %15, float 0xBE74442D00000000, float %.02.i) #7
  %19 = tail call float @llvm.fma.f32(float %15, float 0xBE74442D00000000, float %.02.i)
  %.03.i = select i1 %.not.i, float %19, float %18
  %20 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %15, float 0xBCF84698A0000000, float %.03.i) #7
  %21 = tail call float @llvm.fma.f32(float %15, float 0xBCF84698A0000000, float %.03.i)
  %.04.i = select i1 %.not.i, float %21, float %20
  %22 = tail call float @llvm.nvvm.fabs.ftz.f32(float %v57)
  %23 = tail call float @llvm.nvvm.fabs.f32(float %v57)
  %.06.i = select i1 %.not.i, float %23, float %22
  %24 = fcmp ult float %.06.i, 1.056150e+05
  br i1 %24, label %__nv_cosf.exit, label %__nv_isinff.exit.i.i.i

__nv_isinff.exit.i.i.i:                           ; preds = %__nv_sqrtf.exit
  %25 = fcmp oeq float %.06.i, 0x7FF0000000000000
  br i1 %25, label %__nv_fmul_rn.exit.i.i.i, label %28

__nv_fmul_rn.exit.i.i.i:                          ; preds = %__nv_isinff.exit.i.i.i
  %26 = tail call float @llvm.nvvm.mul.rn.ftz.f(float %v57, float 0.000000e+00) #7
  %27 = tail call float @llvm.nvvm.mul.rn.f(float %v57, float 0.000000e+00) #7
  %.08.i = select i1 %.not.i, float %27, float %26
  br label %__nv_cosf.exit

28:                                               ; preds = %__nv_isinff.exit.i.i.i
  %29 = bitcast float %v57 to i32
  %30 = shl i32 %29, 8
  %31 = or i32 %30, -2147483648
  br label %32

32:                                               ; preds = %28, %32
  %iq.i.i.i.0.i19 = phi i32 [ 0, %28 ], [ %40, %32 ]
  %hi.i.i.i.0.i18 = phi i32 [ 0, %28 ], [ %38, %32 ]
  %33 = zext nneg i32 %iq.i.i.i.0.i19 to i64
  %34 = getelementptr inbounds nuw [6 x i32], ptr addrspace(1) @__cudart_i2opi_f, i64 0, i64 %33
  %35 = load i32, ptr addrspace(1) %34, align 4
  %36 = tail call { i32, i32 } asm "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r"(i32 %35, i32 %31, i32 %hi.i.i.i.0.i18) #8, !srcloc !2
  %37 = extractvalue { i32, i32 } %36, 0
  %38 = extractvalue { i32, i32 } %36, 1
  %39 = getelementptr inbounds nuw [7 x i32], ptr %result.i.i.i.i, i64 0, i64 %33
  store i32 %37, ptr %39, align 4
  %40 = add nuw nsw i32 %iq.i.i.i.0.i19, 1
  %exitcond.not = icmp eq i32 %40, 6
  br i1 %exitcond.not, label %41, label %32, !llvm.loop !3

41:                                               ; preds = %32
  %42 = lshr i32 %29, 23
  %43 = and i32 %42, 224
  %44 = add nsw i32 %43, -128
  %45 = lshr exact i32 %44, 5
  %46 = getelementptr inbounds nuw i8, ptr %result.i.i.i.i, i64 24
  store i32 %38, ptr %46, align 4
  %47 = sub nsw i32 6, %45
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds [7 x i32], ptr %result.i.i.i.i, i64 0, i64 %48
  %50 = load i32, ptr %49, align 4
  %51 = sub nsw i32 5, %45
  %52 = sext i32 %51 to i64
  %53 = getelementptr inbounds [7 x i32], ptr %result.i.i.i.i, i64 0, i64 %52
  %54 = load i32, ptr %53, align 4
  %55 = freeze i32 %54
  %56 = and i32 %29, 260046848
  %.not8.i = icmp eq i32 %56, 0
  br i1 %.not8.i, label %__internal_trig_reduction_slowpath.exit.i.i.i, label %57

57:                                               ; preds = %41
  %58 = sub nsw i32 4, %45
  %59 = sext i32 %58 to i64
  %60 = getelementptr inbounds [7 x i32], ptr %result.i.i.i.i, i64 0, i64 %59
  %61 = load i32, ptr %60, align 4
  %62 = tail call i32 @llvm.fshl.i32(i32 %55, i32 %61, i32 %42)
  br label %__internal_trig_reduction_slowpath.exit.i.i.i

__internal_trig_reduction_slowpath.exit.i.i.i:    ; preds = %57, %41
  %lo.i.i.i.0.i = phi i32 [ %62, %57 ], [ %55, %41 ]
  %63 = tail call i32 @llvm.fshl.i32(i32 %50, i32 %55, i32 %42)
  %64 = lshr i32 %63, 30
  %65 = tail call i32 @llvm.fshl.i32(i32 %63, i32 %lo.i.i.i.0.i, i32 2)
  %66 = shl i32 %lo.i.i.i.0.i, 2
  %67 = lshr i32 %65, 31
  %68 = add nuw nsw i32 %67, %64
  %69 = sub nsw i32 0, %68
  %.not911.i = icmp slt i32 %29, 0
  %spec.select.i = select i1 %.not911.i, i32 %69, i32 %68
  %70 = xor i32 %65, %29
  %.lobit.i = ashr i32 %65, 31
  %hi.i.i.i.2.i = xor i32 %.lobit.i, %65
  %lo.i.i.i.1.i = xor i32 %.lobit.i, %66
  %71 = zext i32 %hi.i.i.i.2.i to i64
  %72 = shl nuw i64 %71, 32
  %73 = zext i32 %lo.i.i.i.1.i to i64
  %74 = or disjoint i64 %72, %73
  %75 = sitofp i64 %74 to double
  %76 = fmul double %75, 0x3BF921FB54442D19
  %77 = fptrunc double %76 to float
  %78 = fneg float %77
  %.not1314.i = icmp slt i32 %70, 0
  %r.i.i.i.0.i = select i1 %.not1314.i, float %78, float %77
  br label %__nv_cosf.exit

__nv_cosf.exit:                                   ; preds = %__nv_sqrtf.exit, %__nv_fmul_rn.exit.i.i.i, %__internal_trig_reduction_slowpath.exit.i.i.i
  %i.i.1.i = phi i32 [ %.01.i, %__nv_sqrtf.exit ], [ 0, %__nv_fmul_rn.exit.i.i.i ], [ %spec.select.i, %__internal_trig_reduction_slowpath.exit.i.i.i ]
  %t.i.i.1.i = phi float [ %.04.i, %__nv_sqrtf.exit ], [ %.08.i, %__nv_fmul_rn.exit.i.i.i ], [ %r.i.i.i.0.i, %__internal_trig_reduction_slowpath.exit.i.i.i ]
  %79 = add i32 %i.i.1.i, 1
  %80 = tail call float @llvm.nvvm.mul.rn.ftz.f(float %t.i.i.1.i, float %t.i.i.1.i) #7
  %81 = tail call float @llvm.nvvm.mul.rn.f(float %t.i.i.1.i, float %t.i.i.1.i) #7
  %.011.i = select i1 %.not.i, float %81, float %80
  %82 = and i32 %i.i.1.i, 1
  %.not15.not.i = icmp eq i32 %82, 0
  %83 = select i1 %.not15.not.i, float 1.000000e+00, float %t.i.i.1.i
  %84 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.011.i, float %83, float 0.000000e+00) #7
  %85 = tail call float @llvm.fma.f32(float %.011.i, float %83, float 0.000000e+00)
  %.012.i = select i1 %.not.i, float %85, float %84
  %86 = tail call float @llvm.nvvm.fma.rn.ftz.f(float 0x3EF9758000000000, float %.011.i, float 0xBF56C0FDA0000000) #7
  %87 = tail call float @llvm.fma.f32(float %.011.i, float 0x3EF9758000000000, float 0xBF56C0FDA0000000)
  %.013.i = select i1 %.not.i, float %87, float %86
  %88 = select i1 %.not15.not.i, float 0xBFDFFFFFE0000000, float 0xBFC5555500000000
  %89 = select i1 %.not15.not.i, float 0x3FA5555760000000, float 0x3F8110BC80000000
  %90 = select i1 %.not15.not.i, float %.013.i, float 0xBF29A82A60000000
  %91 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %90, float %.011.i, float %89) #7
  %92 = tail call float @llvm.fma.f32(float %90, float %.011.i, float %89)
  %.010.i = select i1 %.not.i, float %92, float %91
  %93 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.010.i, float %.011.i, float %88) #7
  %94 = tail call float @llvm.fma.f32(float %.010.i, float %.011.i, float %88)
  %.09.i = select i1 %.not.i, float %94, float %93
  %95 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.09.i, float %.012.i, float %83) #7
  %96 = tail call float @llvm.fma.f32(float %.09.i, float %.012.i, float %83)
  %.05.i = select i1 %.not.i, float %96, float %95
  %97 = and i32 %79, 2
  %.not16.i = icmp eq i32 %97, 0
  %98 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.05.i, float -1.000000e+00, float 0.000000e+00) #7
  %99 = fsub float 0.000000e+00, %96
  %.0.i9 = select i1 %.not.i, float %99, float %98
  %z.i.i.0.i = select i1 %.not16.i, float %.05.i, float %.0.i9
  call void @llvm.lifetime.end.p0(i64 28, ptr nonnull %result.i.i.i.i)
  %v59 = fmul contract float %z.i.i.0.i, 1.270000e+02
  %v60 = fadd contract float %v55, 1.000000e+00
  %v61 = fdiv contract float %v59, %v60
  %v62 = fadd contract float %v61, 1.280000e+02
  %v63 = tail call float @llvm.floor.f32(float %v62) #6
  %v10.i = and i64 %v52.i, 4294967295
  %v14.not.i = icmp samesign uge i64 %v10.i, %v6.i
  %v22.i = mul nuw nsw i64 %v24, %v6.i
  %v23.i11 = add nuw nsw i64 %v22.i, %v10.i
  %v26.not.i = icmp uge i64 %v23.i11, %v1
  %.not = select i1 %v14.not.i, i1 true, i1 %v26.not.i
  %v43.not.not17 = icmp eq ptr %v0, null
  %v43.not.not = select i1 %.not, i1 true, i1 %v43.not.not17
  br i1 %v43.not.not, label %bb10, label %bb6

bb6:                                              ; preds = %__nv_cosf.exit
  %v30.i = getelementptr inbounds nuw { [4 x float] }, ptr %v0, i64 %v23.i11
  %v40.repack5 = getelementptr inbounds nuw i8, ptr %v30.i, i64 12
  %v40.repack3 = getelementptr inbounds nuw i8, ptr %v30.i, i64 8
  %v40.repack1 = getelementptr inbounds nuw i8, ptr %v30.i, i64 4
  store float %v63, ptr %v30.i, align 16
  store float %v63, ptr %v40.repack1, align 4
  store float %v63, ptr %v40.repack3, align 8
  store float 2.550000e+02, ptr %v40.repack5, align 4
  br label %bb10

bb10:                                             ; preds = %__nv_cosf.exit, %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, %bb6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.floor.f32(float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nofree nosync nounwind memory(none)
declare noundef i32 @__nvvm_reflect(ptr noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.f2i.rn.ftz(float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.f2i.rn(float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.ftz.f(float, float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.ftz.f32(float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.f32(float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.mul.rn.ftz.f(float, float) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.mul.rn.f(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #5

attributes #0 = { convergent memory(argmem: write) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nosync nounwind memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { convergent }
attributes #7 = { nounwind }
attributes #8 = { nounwind memory(none) }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
!2 = !{i32 30999, i32 31003, i32 31048, i32 31093}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.unroll.count", i32 1}
