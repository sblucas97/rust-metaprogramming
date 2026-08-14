; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.uint2 = type { i32, i32 }

@llvm.used = appending global [1 x ptr] [ptr @ripple], section "llvm.metadata"
@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1
@__cudart_i2opi_f = internal addrspace(1) global [6 x i32] [i32 1011060801, i32 -614296167, i32 -181084736, i32 -64530479, i32 1313084713, i32 -1560706194], align 4

; Function Attrs: convergent
define ptx_kernel void @ripple(ptr %v0, i64 %v1, i32 %v2, float %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64, i32 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64, i32 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64, i32 } %v5, i32 %v2, 2
  br label %bb0

bb0:                                              ; preds = %entry
  %v7 = phi { ptr, i64, i32 } [ %v6, %entry ]
  %v8 = phi float [ %v3, %entry ]
  %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_ = alloca { ptr, i64, i32 }, align 8
  %v10 = alloca {}, align 1
  %v11 = alloca [4 x float], align 4
  store { ptr, i64, i32 } %v7, ptr %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_, align 8
  %v12 = bitcast ptr %v10 to ptr
  %v13 = bitcast ptr %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_ to ptr
  %v14 = call { i64, i64 } @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_(ptr %v12, ptr %v13) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v15 = extractvalue { i64, i64 } %v14, 0
  %v16 = bitcast i64 %v15 to i64
  %v17 = icmp eq i64 %v16, 1
  br i1 %v17, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  %v18 = icmp eq i64 %v16, 0
  br i1 %v18, label %bb9, label %bb14

bb3:                                              ; preds = %bb1
  %v19 = extractvalue { i64, i64 } %v14, 1
  %v20 = and i64 %v19, 4294967295
  %v21 = trunc i64 %v20 to i32
  %v22 = zext i32 32 to i64
  %v23 = and i64 %v22, 63
  %v24 = lshr i64 %v19, %v23
  %v25 = trunc i64 %v24 to i32
  %v26 = load { ptr, i64, i32 }, ptr %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_, align 8
  %v27 = extractvalue { ptr, i64, i32 } %v26, 2
  %v28 = uitofp i32 %v27 to float
  %v29 = uitofp i32 %v21 to float
  %v30 = fmul contract float 5.000000e-01, %v29
  %v31 = fdiv contract float %v28, 1.500000e+01
  %v32 = fsub contract float %v30, %v31
  %v33 = uitofp i32 %v25 to float
  %v34 = fmul contract float 5.000000e-01, %v33
  %v35 = fsub contract float %v34, %v31
  %v36 = fmul contract float %v32, %v32
  %v37 = fmul contract float %v35, %v35
  %v38 = fadd contract float %v36, %v37
  %v39 = call float @__nv_sqrtf(float %v38) #0
  br label %bb11

bb4:                                              ; preds = %bb13
  %v40 = extractvalue { ptr } %v64, 0
  %v41 = ptrtoint ptr %v40 to i64
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  br i1 %v46, label %bb6, label %bb5

bb5:                                              ; preds = %bb4
  %v47 = icmp eq i64 %v45, 0
  br i1 %v47, label %bb7, label %bb14

bb6:                                              ; preds = %bb4
  %v48 = extractvalue { ptr } %v64, 0
  %v49 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 0
  store float %v63, ptr %v49, align 4
  %v50 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 1
  store float %v63, ptr %v50, align 4
  %v51 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 2
  store float %v63, ptr %v51, align 4
  %v52 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 3
  store float 2.550000e+02, ptr %v52, align 4
  %v53 = load [4 x float], ptr %v11, align 4
  %v54 = insertvalue { [4 x float] } undef, [4 x float] %v53, 0
  store { [4 x float] } %v54, ptr %v48, align 16
  br label %bb8

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  br label %bb10

bb9:                                              ; preds = %bb2
  br label %bb10

bb10:                                             ; preds = %bb9, %bb8
  ret void

bb11:                                             ; preds = %bb3
  %v55 = fdiv contract float %v39, 1.000000e+01
  %v56 = fdiv contract float %v8, 7.000000e+00
  %v57 = fsub contract float %v55, %v56
  %v58 = call float @__nv_cosf(float %v57) #0
  br label %bb12

bb12:                                             ; preds = %bb11
  %v59 = fmul contract float 1.270000e+02, %v58
  %v60 = fadd contract float %v55, 1.000000e+00
  %v61 = fdiv contract float %v59, %v60
  %v62 = fadd contract float 1.280000e+02, %v61
  %v63 = call float @llvm.floor.f32(float %v62) #0
  br label %bb13

bb13:                                             ; preds = %bb12
  %v64 = call { ptr } @_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCslj41F3F2J0U_6ripple4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_(ptr %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_, i64 %v19) #0
  br label %bb4

bb14:                                             ; preds = %bb5, %bb2
  unreachable
}

; Function Attrs: alwaysinline convergent
define { i64, i64 } @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_(ptr %v0, ptr %v1) #1 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load { ptr, i64, i32 }, ptr %v3, align 8
  %v5 = extractvalue { ptr, i64, i32 } %v4, 2
  %v6 = zext i32 %v5 to i64
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb1, label %bb2

bb1:                                              ; preds = %bb0
  %v8 = insertvalue { i64, i64 } undef, i64 0, 0
  %v9 = extractvalue { i64, i64 } %v8, 0
  %v10 = extractvalue { i64, i64 } %v8, 1
  br label %bb21

bb2:                                              ; preds = %bb0
  %v11 = icmp eq i8 0, 1
  %v12 = xor i1 %v11, true
  br i1 %v12, label %bb22, label %bb23

bb3:                                              ; preds = %bb29
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb5

bb4:                                              ; preds = %bb29
  %v14 = insertvalue { i64, i64 } undef, i64 0, 0
  %v15 = extractvalue { i64, i64 } %v14, 0
  %v16 = extractvalue { i64, i64 } %v14, 1
  br label %bb21

bb5:                                              ; preds = %bb3
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb6

bb6:                                              ; preds = %bb5
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb7

bb7:                                              ; preds = %bb6
  %v19 = zext i32 %v13 to i64
  %v20 = zext i32 %v17 to i64
  %v21 = zext i32 %v18 to i64
  %v22 = icmp eq i64 %v20, 0
  br i1 %v22, label %bb32, label %bb30

bb8:                                              ; preds = %bb34
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb9

bb9:                                              ; preds = %bb8
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb10

bb10:                                             ; preds = %bb9
  %v25 = zext i32 %v70 to i64
  %v26 = zext i32 %v23 to i64
  %v27 = zext i32 %v24 to i64
  %v28 = icmp eq i64 %v26, 0
  br i1 %v28, label %bb37, label %bb35

bb11:                                             ; preds = %bb39
  br label %bb16

bb12:                                             ; preds = %bb39
  %v29 = icmp eq i64 %v77, -1
  br i1 %v29, label %bb13, label %bb14

bb13:                                             ; preds = %bb12
  br label %bb16

bb14:                                             ; preds = %bb12
  %v30 = icmp uge i64 %v77, %v6
  %v31 = xor i1 %v30, true
  br i1 %v31, label %bb17, label %bb15

bb15:                                             ; preds = %bb14
  br label %bb16

bb16:                                             ; preds = %bb15, %bb13, %bb11
  %v32 = insertvalue { i64, i64 } undef, i64 0, 0
  %v33 = extractvalue { i64, i64 } %v32, 0
  %v34 = extractvalue { i64, i64 } %v32, 1
  br label %bb20

bb17:                                             ; preds = %bb14
  %v35 = icmp ugt i64 %v69, 4294967295
  %v36 = xor i1 %v35, true
  br i1 %v36, label %bb19, label %bb18

bb18:                                             ; preds = %bb17
  %v37 = insertvalue { i64, i64 } undef, i64 0, 0
  %v38 = extractvalue { i64, i64 } %v37, 0
  %v39 = extractvalue { i64, i64 } %v37, 1
  br label %bb20

bb19:                                             ; preds = %bb17
  %v40 = zext i32 32 to i64
  %v41 = and i64 %v40, 63
  %v42 = shl i64 %v69, %v41
  %v43 = or i64 %v42, %v77
  %v44 = icmp eq i64 %v43, -1
  br i1 %v44, label %bb40, label %bb41

bb20:                                             ; preds = %bb18, %bb16
  %v45 = phi i64 [ %v33, %bb16 ], [ %v38, %bb18 ]
  %v46 = phi i64 [ %v34, %bb16 ], [ %v39, %bb18 ]
  %v47 = insertvalue { i64, i64 } undef, i64 %v45, 0
  %v48 = insertvalue { i64, i64 } %v47, i64 %v46, 1
  %v49 = extractvalue { i64, i64 } %v48, 0
  %v50 = extractvalue { i64, i64 } %v48, 1
  br label %bb21

bb21:                                             ; preds = %bb41, %bb20, %bb4, %bb1
  %v51 = phi i64 [ %v9, %bb1 ], [ %v15, %bb4 ], [ %v49, %bb20 ], [ %v82, %bb41 ]
  %v52 = phi i64 [ %v10, %bb1 ], [ %v16, %bb4 ], [ %v50, %bb20 ], [ %v83, %bb41 ]
  %v53 = insertvalue { i64, i64 } undef, i64 %v51, 0
  %v54 = insertvalue { i64, i64 } %v53, i64 %v52, 1
  ret { i64, i64 } %v54

bb22:                                             ; preds = %bb2
  %v55 = icmp eq i8 0, 2
  %v56 = xor i1 %v55, true
  br i1 %v56, label %bb24, label %bb23

bb23:                                             ; preds = %bb22, %bb2
  br label %bb29

bb24:                                             ; preds = %bb22
  %v57 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb25

bb25:                                             ; preds = %bb24
  %v58 = icmp eq i32 %v57, 1
  br i1 %v58, label %bb26, label %bb27

bb26:                                             ; preds = %bb25
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb28

bb27:                                             ; preds = %bb25
  br label %bb29

bb28:                                             ; preds = %bb26
  %v60 = icmp eq i32 %v59, 1
  br label %bb29

bb29:                                             ; preds = %bb28, %bb27, %bb23
  %v61 = phi i1 [ true, %bb23 ], [ false, %bb27 ], [ %v60, %bb28 ]
  %v62 = xor i1 %v61, true
  br i1 %v62, label %bb4, label %bb3

bb30:                                             ; preds = %bb7
  %v63 = sub i64 -1, %v21
  %v64 = udiv i64 %v63, %v20
  %v65 = icmp ugt i64 %v19, %v64
  %v66 = xor i1 %v65, true
  br i1 %v66, label %bb33, label %bb31

bb31:                                             ; preds = %bb30
  br label %bb32

bb32:                                             ; preds = %bb31, %bb7
  br label %bb34

bb33:                                             ; preds = %bb30
  %v67 = mul i64 %v19, %v20
  %v68 = add i64 %v67, %v21
  br label %bb34

bb34:                                             ; preds = %bb33, %bb32
  %v69 = phi i64 [ -1, %bb32 ], [ %v68, %bb33 ]
  %v70 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb8

bb35:                                             ; preds = %bb10
  %v71 = sub i64 -1, %v27
  %v72 = udiv i64 %v71, %v26
  %v73 = icmp ugt i64 %v25, %v72
  %v74 = xor i1 %v73, true
  br i1 %v74, label %bb38, label %bb36

bb36:                                             ; preds = %bb35
  br label %bb37

bb37:                                             ; preds = %bb36, %bb10
  br label %bb39

bb38:                                             ; preds = %bb35
  %v75 = mul i64 %v25, %v26
  %v76 = add i64 %v75, %v27
  br label %bb39

bb39:                                             ; preds = %bb38, %bb37
  %v77 = phi i64 [ -1, %bb37 ], [ %v76, %bb38 ]
  %v78 = icmp eq i64 %v69, -1
  br i1 %v78, label %bb11, label %bb12

bb40:                                             ; preds = %bb19
  br label %bb41

bb41:                                             ; preds = %bb40, %bb19
  %v79 = phi i64 [ %v43, %bb19 ], [ -1, %bb40 ]
  %v80 = insertvalue { i64, i64 } undef, i64 1, 0
  %v81 = insertvalue { i64, i64 } %v80, i64 %v79, 1
  %v82 = extractvalue { i64, i64 } %v81, 0
  %v83 = extractvalue { i64, i64 } %v81, 1
  br label %bb21
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.floor.f32(float) #2

; Function Attrs: convergent
define { ptr } @_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCslj41F3F2J0U_6ripple4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_(ptr %v0, i64 %v1) #0 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = icmp eq i64 16, 0
  %v5 = xor i1 %v4, true
  br i1 %v5, label %bb1, label %bb3

bb1:                                              ; preds = %bb0
  %v6 = icmp eq i64 %v3, -1
  br i1 %v6, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  %v7 = zext i32 32 to i64
  %v8 = and i64 %v7, 63
  %v9 = lshr i64 %v3, %v8
  %v10 = and i64 %v3, 4294967295
  %v11 = load { ptr, i64, i32 }, ptr %v2, align 8
  %v12 = extractvalue { ptr, i64, i32 } %v11, 2
  %v13 = zext i32 %v12 to i64
  %v14 = icmp uge i64 %v10, %v13
  %v15 = xor i1 %v14, true
  br i1 %v15, label %bb5, label %bb4

bb3:                                              ; preds = %bb1, %bb0
  %v16 = inttoptr i64 0 to ptr
  %v17 = insertvalue { ptr } undef, ptr %v16, 0
  %v18 = extractvalue { ptr } %v17, 0
  br label %bb9

bb4:                                              ; preds = %bb2
  %v19 = inttoptr i64 0 to ptr
  %v20 = insertvalue { ptr } undef, ptr %v19, 0
  %v21 = extractvalue { ptr } %v20, 0
  br label %bb9

bb5:                                              ; preds = %bb2
  %v22 = mul i64 %v9, %v13
  %v23 = add i64 %v22, %v10
  %v24 = load { ptr, i64, i32 }, ptr %v2, align 8
  %v25 = extractvalue { ptr, i64, i32 } %v24, 1
  %v26 = icmp ult i64 %v23, %v25
  %v27 = xor i1 %v26, true
  br i1 %v27, label %bb7, label %bb6

bb6:                                              ; preds = %bb5
  %v28 = load { ptr, i64, i32 }, ptr %v2, align 8
  %v29 = extractvalue { ptr, i64, i32 } %v28, 0
  %v30 = getelementptr inbounds { [4 x float] }, ptr %v29, i64 %v23
  %v31 = insertvalue { ptr } undef, ptr %v30, 0
  %v32 = extractvalue { ptr } %v31, 0
  br label %bb8

bb7:                                              ; preds = %bb5
  %v33 = inttoptr i64 0 to ptr
  %v34 = insertvalue { ptr } undef, ptr %v33, 0
  %v35 = extractvalue { ptr } %v34, 0
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  %v36 = phi ptr [ %v32, %bb6 ], [ %v35, %bb7 ]
  %v37 = insertvalue { ptr } undef, ptr %v36, 0
  %v38 = extractvalue { ptr } %v37, 0
  br label %bb9

bb9:                                              ; preds = %bb8, %bb4, %bb3
  %v39 = phi ptr [ %v18, %bb3 ], [ %v21, %bb4 ], [ %v38, %bb8 ]
  %v40 = insertvalue { ptr } undef, ptr %v39, 0
  ret { ptr } %v40
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: alwaysinline nounwind
define internal float @__nv_sqrtf(float %x) #3 {
  %1 = call i32 @__nvvm_reflect(ptr @.str) #6
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %10

3:                                                ; preds = %0
  %4 = call i32 @__nvvm_reflect(ptr @.str.2) #6
  %5 = icmp ne i32 %4, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %3
  %7 = call float @llvm.nvvm.sqrt.rn.ftz.f(float %x) #6
  br label %__nvvm_sqrt_f.exit

8:                                                ; preds = %3
  %9 = call float @llvm.nvvm.sqrt.approx.ftz.f(float %x) #6
  br label %__nvvm_sqrt_f.exit

10:                                               ; preds = %0
  %11 = call i32 @__nvvm_reflect(ptr @.str.2) #6
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %15

13:                                               ; preds = %10
  %14 = call float @llvm.nvvm.sqrt.rn.f(float %x) #6
  br label %__nvvm_sqrt_f.exit

15:                                               ; preds = %10
  %16 = call float @llvm.nvvm.sqrt.approx.f(float %x) #6
  br label %__nvvm_sqrt_f.exit

__nvvm_sqrt_f.exit:                               ; preds = %6, %8, %13, %15
  %.0 = phi float [ %7, %6 ], [ %9, %8 ], [ %14, %13 ], [ %16, %15 ]
  ret float %.0
}

declare i32 @__nvvm_reflect(ptr) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #5

; Function Attrs: alwaysinline nounwind
define internal float @__nv_cosf(float %a) #3 {
  %result.i.i.i = alloca [7 x i32], align 4
  %1 = fmul float %a, 0x3FE45F3060000000
  %2 = call i32 @__nvvm_reflect(ptr @.str) #6
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %6

4:                                                ; preds = %0
  %5 = call i32 @llvm.nvvm.f2i.rn.ftz(float %1) #6
  br label %__nv_float2int_rn.exit.i.i

6:                                                ; preds = %0
  %7 = call i32 @llvm.nvvm.f2i.rn(float %1) #6
  br label %__nv_float2int_rn.exit.i.i

__nv_float2int_rn.exit.i.i:                       ; preds = %6, %4
  %.01 = phi i32 [ %5, %4 ], [ %7, %6 ]
  %8 = sitofp i32 %.01 to float
  %9 = call i32 @__nvvm_reflect(ptr @.str) #6
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %13

11:                                               ; preds = %__nv_float2int_rn.exit.i.i
  %12 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBFF921FB40000000, float %a) #6
  br label %__nv_fmaf_rn.exit.i.i

13:                                               ; preds = %__nv_float2int_rn.exit.i.i
  %14 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBFF921FB40000000, float %a) #6
  br label %__nv_fmaf_rn.exit.i.i

__nv_fmaf_rn.exit.i.i:                            ; preds = %13, %11
  %.02 = phi float [ %12, %11 ], [ %14, %13 ]
  %15 = call i32 @__nvvm_reflect(ptr @.str) #6
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %17, label %19

17:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %18 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBE74442D00000000, float %.02) #6
  br label %__nv_fmaf_rn.exit1.i.i

19:                                               ; preds = %__nv_fmaf_rn.exit.i.i
  %20 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBE74442D00000000, float %.02) #6
  br label %__nv_fmaf_rn.exit1.i.i

__nv_fmaf_rn.exit1.i.i:                           ; preds = %19, %17
  %.03 = phi float [ %18, %17 ], [ %20, %19 ]
  %21 = call i32 @__nvvm_reflect(ptr @.str) #6
  %22 = icmp ne i32 %21, 0
  br i1 %22, label %23, label %25

23:                                               ; preds = %__nv_fmaf_rn.exit1.i.i
  %24 = call float @llvm.nvvm.fma.rn.ftz.f(float %8, float 0xBCF84698A0000000, float %.03) #6
  br label %__nv_fmaf_rn.exit2.i.i

25:                                               ; preds = %__nv_fmaf_rn.exit1.i.i
  %26 = call float @llvm.nvvm.fma.rn.f(float %8, float 0xBCF84698A0000000, float %.03) #6
  br label %__nv_fmaf_rn.exit2.i.i

__nv_fmaf_rn.exit2.i.i:                           ; preds = %25, %23
  %.04 = phi float [ %24, %23 ], [ %26, %25 ]
  %27 = call i32 @__nvvm_reflect(ptr @.str) #6
  %28 = icmp ne i32 %27, 0
  br i1 %28, label %29, label %31

29:                                               ; preds = %__nv_fmaf_rn.exit2.i.i
  %30 = call float @llvm.nvvm.fabs.ftz.f32(float %a)
  br label %__nv_fabsf.exit.i.i

31:                                               ; preds = %__nv_fmaf_rn.exit2.i.i
  %32 = call float @llvm.nvvm.fabs.f32(float %a)
  br label %__nv_fabsf.exit.i.i

__nv_fabsf.exit.i.i:                              ; preds = %31, %29
  %.06 = phi float [ %30, %29 ], [ %32, %31 ]
  %33 = fcmp oge float %.06, 1.056150e+05
  br i1 %33, label %34, label %__internal_trig_reduction_kernel.exit.i

34:                                               ; preds = %__nv_fabsf.exit.i.i
  %35 = call i32 @__nvvm_reflect(ptr @.str) #6
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %39

37:                                               ; preds = %34
  %38 = call float @llvm.nvvm.fabs.ftz.f32(float %a)
  br label %__nv_isinff.exit.i.i

39:                                               ; preds = %34
  %40 = call float @llvm.nvvm.fabs.f32(float %a)
  br label %__nv_isinff.exit.i.i

__nv_isinff.exit.i.i:                             ; preds = %39, %37
  %.07 = phi float [ %38, %37 ], [ %40, %39 ]
  %41 = bitcast i32 2139095040 to float
  %42 = fcmp oeq float %.07, %41
  %43 = select i1 %42, i32 1, i32 0
  br i1 %42, label %44, label %51

44:                                               ; preds = %__nv_isinff.exit.i.i
  %45 = call i32 @__nvvm_reflect(ptr @.str) #6
  %46 = icmp ne i32 %45, 0
  br i1 %46, label %47, label %49

47:                                               ; preds = %44
  %48 = call float @llvm.nvvm.mul.rn.ftz.f(float %a, float 0.000000e+00) #6
  br label %__nv_fmul_rn.exit.i.i

49:                                               ; preds = %44
  %50 = call float @llvm.nvvm.mul.rn.f(float %a, float 0.000000e+00) #6
  br label %__nv_fmul_rn.exit.i.i

__nv_fmul_rn.exit.i.i:                            ; preds = %49, %47
  %.08 = phi float [ %48, %47 ], [ %50, %49 ]
  br label %127

51:                                               ; preds = %__nv_isinff.exit.i.i
  %52 = bitcast float %a to i32
  %53 = and i32 %52, -2147483648
  %54 = lshr i32 %52, 23
  %55 = and i32 %54, 255
  %56 = sub i32 %55, 128
  %57 = shl i32 %52, 8
  %58 = or i32 %57, -2147483648
  %59 = lshr i32 %56, 5
  %60 = sub i32 4, %59
  br label %61

61:                                               ; preds = %63, %51
  %hi.i.i.i.0 = phi i32 [ 0, %51 ], [ %71, %63 ]
  %iq.i.i.i.0 = phi i32 [ 0, %51 ], [ %74, %63 ]
  %62 = icmp slt i32 %iq.i.i.i.0, 6
  br i1 %62, label %63, label %75

63:                                               ; preds = %61
  %64 = sext i32 %iq.i.i.i.0 to i64
  %65 = getelementptr inbounds [6 x i32], ptr addrspace(1) @__cudart_i2opi_f, i64 0, i64 %64
  %66 = load i32, ptr addrspace(1) %65, align 4
  %67 = call { i32, i32 } asm "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r"(i32 %66, i32 %58, i32 %hi.i.i.i.0) #7, !srcloc !2
  %68 = extractvalue { i32, i32 } %67, 0
  %69 = extractvalue { i32, i32 } %67, 1
  %insert = insertvalue %struct.uint2 undef, i32 %68, 0
  %insert25 = insertvalue %struct.uint2 %insert, i32 %69, 1
  %70 = extractvalue %struct.uint2 %insert25, 0
  %71 = extractvalue %struct.uint2 %insert25, 1
  %72 = sext i32 %iq.i.i.i.0 to i64
  %73 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %72
  store i32 %70, ptr %73, align 4
  %74 = add nsw i32 %iq.i.i.i.0, 1
  br label %61, !llvm.loop !3

75:                                               ; preds = %61
  %76 = sext i32 %iq.i.i.i.0 to i64
  %77 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %76
  store i32 %hi.i.i.i.0, ptr %77, align 4
  %78 = and i32 %56, 31
  %79 = add i32 %60, 2
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %80
  %82 = load i32, ptr %81, align 4
  %83 = add i32 %60, 1
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %84
  %86 = load i32, ptr %85, align 4
  %87 = icmp ne i32 %78, 0
  br i1 %87, label %88, label %99

88:                                               ; preds = %75
  %89 = sub i32 32, %78
  %90 = shl i32 %82, %78
  %91 = lshr i32 %86, %89
  %92 = add i32 %90, %91
  %93 = shl i32 %86, %78
  %94 = sext i32 %60 to i64
  %95 = getelementptr inbounds [7 x i32], ptr %result.i.i.i, i64 0, i64 %94
  %96 = load i32, ptr %95, align 4
  %97 = lshr i32 %96, %89
  %98 = add i32 %93, %97
  br label %99

99:                                               ; preds = %88, %75
  %hi.i.i.i.1 = phi i32 [ %92, %88 ], [ %82, %75 ]
  %lo.i.i.i.0 = phi i32 [ %98, %88 ], [ %86, %75 ]
  %100 = lshr i32 %hi.i.i.i.1, 30
  %101 = shl i32 %hi.i.i.i.1, 2
  %102 = lshr i32 %lo.i.i.i.0, 30
  %103 = add i32 %101, %102
  %104 = shl i32 %lo.i.i.i.0, 2
  %105 = lshr i32 %103, 31
  %106 = add i32 %100, %105
  %107 = icmp ne i32 %53, 0
  br i1 %107, label %108, label %110

108:                                              ; preds = %99
  %109 = sub i32 0, %106
  br label %110

110:                                              ; preds = %108, %99
  %q.i.i.i.0 = phi i32 [ %109, %108 ], [ %106, %99 ]
  %111 = icmp ne i32 %105, 0
  br i1 %111, label %112, label %116

112:                                              ; preds = %110
  %113 = xor i32 %103, -1
  %114 = xor i32 %104, -1
  %115 = xor i32 %53, -2147483648
  br label %116

116:                                              ; preds = %112, %110
  %s.i.i.i.0 = phi i32 [ %115, %112 ], [ %53, %110 ]
  %hi.i.i.i.2 = phi i32 [ %113, %112 ], [ %103, %110 ]
  %lo.i.i.i.1 = phi i32 [ %114, %112 ], [ %104, %110 ]
  %117 = zext i32 %hi.i.i.i.2 to i64
  %118 = shl i64 %117, 32
  %119 = zext i32 %lo.i.i.i.1 to i64
  %120 = or i64 %118, %119
  %121 = sitofp i64 %120 to double
  %122 = fmul double %121, 0x3BF921FB54442D19
  %123 = fptrunc double %122 to float
  %124 = icmp ne i32 %s.i.i.i.0, 0
  br i1 %124, label %125, label %__internal_trig_reduction_slowpath.exit.i.i

125:                                              ; preds = %116
  %126 = fsub float -0.000000e+00, %123
  br label %__internal_trig_reduction_slowpath.exit.i.i

__internal_trig_reduction_slowpath.exit.i.i:      ; preds = %125, %116
  %r.i.i.i.0 = phi float [ %126, %125 ], [ %123, %116 ]
  br label %127

127:                                              ; preds = %__internal_trig_reduction_slowpath.exit.i.i, %__nv_fmul_rn.exit.i.i
  %i.i.0 = phi i32 [ 0, %__nv_fmul_rn.exit.i.i ], [ %q.i.i.i.0, %__internal_trig_reduction_slowpath.exit.i.i ]
  %t.i.i.0 = phi float [ %.08, %__nv_fmul_rn.exit.i.i ], [ %r.i.i.i.0, %__internal_trig_reduction_slowpath.exit.i.i ]
  br label %__internal_trig_reduction_kernel.exit.i

__internal_trig_reduction_kernel.exit.i:          ; preds = %127, %__nv_fabsf.exit.i.i
  %i.i.1 = phi i32 [ %i.i.0, %127 ], [ %.01, %__nv_fabsf.exit.i.i ]
  %t.i.i.1 = phi float [ %t.i.i.0, %127 ], [ %.04, %__nv_fabsf.exit.i.i ]
  %128 = add i32 %i.i.1, 1
  %129 = call i32 @__nvvm_reflect(ptr @.str) #6
  %130 = icmp ne i32 %129, 0
  br i1 %130, label %131, label %133

131:                                              ; preds = %__internal_trig_reduction_kernel.exit.i
  %132 = call float @llvm.nvvm.mul.rn.ftz.f(float %t.i.i.1, float %t.i.i.1) #6
  br label %__nv_fmul_rn.exit.i2.i

133:                                              ; preds = %__internal_trig_reduction_kernel.exit.i
  %134 = call float @llvm.nvvm.mul.rn.f(float %t.i.i.1, float %t.i.i.1) #6
  br label %__nv_fmul_rn.exit.i2.i

__nv_fmul_rn.exit.i2.i:                           ; preds = %133, %131
  %.011 = phi float [ %132, %131 ], [ %134, %133 ]
  %135 = and i32 %128, 1
  %136 = icmp ne i32 %135, 0
  br i1 %136, label %137, label %138

137:                                              ; preds = %__nv_fmul_rn.exit.i2.i
  br label %139

138:                                              ; preds = %__nv_fmul_rn.exit.i2.i
  br label %139

139:                                              ; preds = %138, %137
  %140 = phi float [ 1.000000e+00, %137 ], [ %t.i.i.1, %138 ]
  %141 = call i32 @__nvvm_reflect(ptr @.str) #6
  %142 = icmp ne i32 %141, 0
  br i1 %142, label %143, label %145

143:                                              ; preds = %139
  %144 = call float @llvm.nvvm.fma.rn.ftz.f(float %.011, float %140, float 0.000000e+00) #6
  br label %__internal_fmad.exit.i.i

145:                                              ; preds = %139
  %146 = call float @llvm.nvvm.fma.rn.f(float %.011, float %140, float 0.000000e+00) #6
  br label %__internal_fmad.exit.i.i

__internal_fmad.exit.i.i:                         ; preds = %145, %143
  %.012 = phi float [ %144, %143 ], [ %146, %145 ]
  %147 = and i32 %128, 1
  %148 = icmp ne i32 %147, 0
  br i1 %148, label %149, label %156

149:                                              ; preds = %__internal_fmad.exit.i.i
  %150 = call i32 @__nvvm_reflect(ptr @.str) #6
  %151 = icmp ne i32 %150, 0
  br i1 %151, label %152, label %154

152:                                              ; preds = %149
  %153 = call float @llvm.nvvm.fma.rn.ftz.f(float 0x3EF9758000000000, float %.011, float 0xBF56C0FDA0000000) #6
  br label %__internal_fmad.exit1.i.i

154:                                              ; preds = %149
  %155 = call float @llvm.nvvm.fma.rn.f(float 0x3EF9758000000000, float %.011, float 0xBF56C0FDA0000000) #6
  br label %__internal_fmad.exit1.i.i

__internal_fmad.exit1.i.i:                        ; preds = %154, %152
  %.013 = phi float [ %153, %152 ], [ %155, %154 ]
  br label %157

156:                                              ; preds = %__internal_fmad.exit.i.i
  br label %157

157:                                              ; preds = %156, %__internal_fmad.exit1.i.i
  %158 = phi float [ %.013, %__internal_fmad.exit1.i.i ], [ 0xBF29A82A60000000, %156 ]
  %159 = and i32 %128, 1
  %160 = icmp ne i32 %159, 0
  %161 = select i1 %160, float 0x3FA5555760000000, float 0x3F8110BC80000000
  %162 = call i32 @__nvvm_reflect(ptr @.str) #6
  %163 = icmp ne i32 %162, 0
  br i1 %163, label %164, label %166

164:                                              ; preds = %157
  %165 = call float @llvm.nvvm.fma.rn.ftz.f(float %158, float %.011, float %161) #6
  br label %__internal_fmad.exit2.i.i

166:                                              ; preds = %157
  %167 = call float @llvm.nvvm.fma.rn.f(float %158, float %.011, float %161) #6
  br label %__internal_fmad.exit2.i.i

__internal_fmad.exit2.i.i:                        ; preds = %166, %164
  %.010 = phi float [ %165, %164 ], [ %167, %166 ]
  %168 = and i32 %128, 1
  %169 = icmp ne i32 %168, 0
  %170 = select i1 %169, float 0xBFDFFFFFE0000000, float 0xBFC5555500000000
  %171 = call i32 @__nvvm_reflect(ptr @.str) #6
  %172 = icmp ne i32 %171, 0
  br i1 %172, label %173, label %175

173:                                              ; preds = %__internal_fmad.exit2.i.i
  %174 = call float @llvm.nvvm.fma.rn.ftz.f(float %.010, float %.011, float %170) #6
  br label %__internal_fmad.exit3.i.i

175:                                              ; preds = %__internal_fmad.exit2.i.i
  %176 = call float @llvm.nvvm.fma.rn.f(float %.010, float %.011, float %170) #6
  br label %__internal_fmad.exit3.i.i

__internal_fmad.exit3.i.i:                        ; preds = %175, %173
  %.09 = phi float [ %174, %173 ], [ %176, %175 ]
  %177 = call i32 @__nvvm_reflect(ptr @.str) #6
  %178 = icmp ne i32 %177, 0
  br i1 %178, label %179, label %181

179:                                              ; preds = %__internal_fmad.exit3.i.i
  %180 = call float @llvm.nvvm.fma.rn.ftz.f(float %.09, float %.012, float %140) #6
  br label %__internal_fmad.exit4.i.i

181:                                              ; preds = %__internal_fmad.exit3.i.i
  %182 = call float @llvm.nvvm.fma.rn.f(float %.09, float %.012, float %140) #6
  br label %__internal_fmad.exit4.i.i

__internal_fmad.exit4.i.i:                        ; preds = %181, %179
  %.05 = phi float [ %180, %179 ], [ %182, %181 ]
  %183 = and i32 %128, 2
  %184 = icmp ne i32 %183, 0
  br i1 %184, label %185, label %__internal_accurate_cosf.exit

185:                                              ; preds = %__internal_fmad.exit4.i.i
  %186 = call i32 @__nvvm_reflect(ptr @.str) #6
  %187 = icmp ne i32 %186, 0
  br i1 %187, label %188, label %190

188:                                              ; preds = %185
  %189 = call float @llvm.nvvm.fma.rn.ftz.f(float %.05, float -1.000000e+00, float 0.000000e+00) #6
  br label %__internal_fmad.exit5.i.i

190:                                              ; preds = %185
  %191 = call float @llvm.nvvm.fma.rn.f(float %.05, float -1.000000e+00, float 0.000000e+00) #6
  br label %__internal_fmad.exit5.i.i

__internal_fmad.exit5.i.i:                        ; preds = %190, %188
  %.0 = phi float [ %189, %188 ], [ %191, %190 ]
  br label %__internal_accurate_cosf.exit

__internal_accurate_cosf.exit:                    ; preds = %__internal_fmad.exit4.i.i, %__internal_fmad.exit5.i.i
  %z.i.i.0 = phi float [ %.0, %__internal_fmad.exit5.i.i ], [ %.05, %__internal_fmad.exit4.i.i ]
  ret float %z.i.i.0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.f2i.rn.ftz(float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.f2i.rn(float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.ftz.f(float, float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.f(float, float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.ftz.f32(float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.f32(float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.mul.rn.ftz.f(float, float) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.mul.rn.f(float, float) #2

attributes #0 = { convergent }
attributes #1 = { alwaysinline convergent }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { alwaysinline nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #6 = { nounwind }
attributes #7 = { nounwind memory(none) }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
!2 = !{i32 30999, i32 31003, i32 31048, i32 31093}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.unroll.count", i32 1}
