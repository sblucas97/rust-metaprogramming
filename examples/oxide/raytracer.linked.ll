; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @raytracing], section "llvm.metadata"
@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1

; Function Attrs: convergent
define ptx_kernel void @raytracing(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64, i32 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64, i32 } %v7, i64 %v3, 1
  %v9 = insertvalue { ptr, i64, i32 } %v8, i32 %v4, 2
  br label %bb0

bb0:                                              ; preds = %entry
  %v10 = phi { ptr, i64 } [ %v6, %entry ]
  %v11 = phi { ptr, i64, i32 } [ %v9, %entry ]
  %__cuda_oxide_local_x3209323409696d616765094469736a6f696e74536c696365_ = alloca { ptr, i64, i32 }, align 8
  %v13 = alloca {}, align 1
  %v14 = alloca [4 x float], align 4
  store { ptr, i64, i32 } %v11, ptr %__cuda_oxide_local_x3209323409696d616765094469736a6f696e74536c696365_, align 8
  %v15 = bitcast ptr %v13 to ptr
  %v16 = bitcast ptr %__cuda_oxide_local_x3209323409696d616765094469736a6f696e74536c696365_ to ptr
  %v17 = call { i64, i64 } @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_(ptr %v15, ptr %v16) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v18 = extractvalue { i64, i64 } %v17, 0
  %v19 = bitcast i64 %v18 to i64
  %v20 = icmp eq i64 %v19, 1
  br i1 %v20, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  %v21 = icmp eq i64 %v19, 0
  br i1 %v21, label %bb26, label %bb5

bb3:                                              ; preds = %bb1
  %v22 = extractvalue { i64, i64 } %v17, 1
  %v23 = load { ptr, i64, i32 }, ptr %__cuda_oxide_local_x3209323409696d616765094469736a6f696e74536c696365_, align 8
  %v24 = extractvalue { ptr, i64, i32 } %v23, 2
  %v25 = uitofp i32 %v24 to float
  %v26 = and i64 %v22, 4294967295
  %v27 = trunc i64 %v26 to i32
  %v28 = zext i32 32 to i64
  %v29 = and i64 %v28, 63
  %v30 = lshr i64 %v22, %v29
  %v31 = trunc i64 %v30 to i32
  %v32 = uitofp i32 %v27 to float
  %v33 = fdiv contract float %v25, 2.000000e+00
  %v34 = fsub contract float %v32, %v33
  %v35 = uitofp i32 %v31 to float
  %v36 = fsub contract float %v35, %v33
  br label %bb4

bb4:                                              ; preds = %bb20, %bb3
  %v37 = phi float [ 0.000000e+00, %bb3 ], [ %v99, %bb20 ]
  %v38 = phi float [ 0.000000e+00, %bb3 ], [ %v100, %bb20 ]
  %v39 = phi float [ 0.000000e+00, %bb3 ], [ %v101, %bb20 ]
  %v40 = phi float [ -9.999900e+04, %bb3 ], [ %v102, %bb20 ]
  %v41 = phi i64 [ 0, %bb3 ], [ %v129, %bb20 ]
  %v42 = icmp ult i64 %v41, 20
  %v43 = xor i1 %v42, true
  br i1 %v43, label %bb29, label %bb28

bb5:                                              ; preds = %bb31, %bb22, %bb2
  unreachable

bb6:                                              ; preds = %bb31
  %v44 = extractvalue { i64, i64 } %v133, 1
  %v45 = mul i64 %v44, 7
  %v46 = add i64 %v45, 3
  %v47 = extractvalue { ptr, i64 } %v10, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb8, label %bb34

bb7:                                              ; preds = %bb30
  %v49 = call { ptr } @_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCs6HixiQUBNAA_9raytracer4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_(ptr %__cuda_oxide_local_x3209323409696d616765094469736a6f696e74536c696365_, i64 %v22) #0
  br label %bb21

bb8:                                              ; preds = %bb6
  %v50 = extractvalue { ptr, i64 } %v10, 0
  %v51 = getelementptr inbounds float, ptr %v50, i64 %v46
  %v52 = load float, ptr %v51, align 4
  %v53 = add i64 %v45, 4
  %v54 = icmp ult i64 %v53, %v47
  br i1 %v54, label %bb9, label %bb35

bb9:                                              ; preds = %bb8
  %v55 = extractvalue { ptr, i64 } %v10, 0
  %v56 = getelementptr inbounds float, ptr %v55, i64 %v53
  %v57 = load float, ptr %v56, align 4
  %v58 = fsub contract float %v34, %v57
  %v59 = add i64 %v45, 5
  %v60 = icmp ult i64 %v59, %v47
  br i1 %v60, label %bb10, label %bb36

bb10:                                             ; preds = %bb9
  %v61 = extractvalue { ptr, i64 } %v10, 0
  %v62 = getelementptr inbounds float, ptr %v61, i64 %v59
  %v63 = load float, ptr %v62, align 4
  %v64 = fsub contract float %v36, %v63
  %v65 = fmul contract float %v58, %v58
  %v66 = fmul contract float %v64, %v64
  %v67 = fadd contract float %v65, %v66
  %v68 = fmul contract float %v52, %v52
  %v69 = fcmp olt float %v67, %v68
  %v70 = xor i1 %v69, true
  br i1 %v70, label %bb13, label %bb11

bb11:                                             ; preds = %bb10
  %v71 = fsub contract float %v68, %v65
  %v72 = fsub contract float %v71, %v66
  %v73 = call float @__nv_sqrtf(float %v72) #0
  br label %bb32

bb12:                                             ; preds = %bb33
  %v74 = extractvalue { ptr, i64 } %v10, 0
  %v75 = getelementptr inbounds float, ptr %v74, i64 %v140
  %v76 = load float, ptr %v75, align 4
  %v77 = fadd contract float %v73, %v76
  br label %bb14

bb13:                                             ; preds = %bb10
  br label %bb14

bb14:                                             ; preds = %bb13, %bb12
  %v78 = phi float [ %v139, %bb12 ], [ 0.000000e+00, %bb13 ]
  %v79 = phi float [ %v77, %bb12 ], [ -9.999900e+04, %bb13 ]
  %v80 = fcmp ogt float %v79, %v40
  %v81 = xor i1 %v80, true
  br i1 %v81, label %bb19, label %bb15

bb15:                                             ; preds = %bb14
  %v82 = icmp ult i64 %v45, %v47
  br i1 %v82, label %bb16, label %bb37

bb16:                                             ; preds = %bb15
  %v83 = extractvalue { ptr, i64 } %v10, 0
  %v84 = getelementptr inbounds float, ptr %v83, i64 %v45
  %v85 = load float, ptr %v84, align 4
  %v86 = fmul contract float %v85, %v78
  %v87 = add i64 %v45, 1
  %v88 = icmp ult i64 %v87, %v47
  br i1 %v88, label %bb17, label %bb38

bb17:                                             ; preds = %bb16
  %v89 = extractvalue { ptr, i64 } %v10, 0
  %v90 = getelementptr inbounds float, ptr %v89, i64 %v87
  %v91 = load float, ptr %v90, align 4
  %v92 = fmul contract float %v91, %v78
  %v93 = add i64 %v45, 2
  %v94 = icmp ult i64 %v93, %v47
  br i1 %v94, label %bb18, label %bb39

bb18:                                             ; preds = %bb17
  %v95 = extractvalue { ptr, i64 } %v10, 0
  %v96 = getelementptr inbounds float, ptr %v95, i64 %v93
  %v97 = load float, ptr %v96, align 4
  %v98 = fmul contract float %v97, %v78
  br label %bb20

bb19:                                             ; preds = %bb14
  br label %bb20

bb20:                                             ; preds = %bb19, %bb18
  %v99 = phi float [ %v86, %bb18 ], [ %v37, %bb19 ]
  %v100 = phi float [ %v92, %bb18 ], [ %v38, %bb19 ]
  %v101 = phi float [ %v98, %bb18 ], [ %v39, %bb19 ]
  %v102 = phi float [ %v79, %bb18 ], [ %v40, %bb19 ]
  br label %bb4

bb21:                                             ; preds = %bb7
  %v103 = extractvalue { ptr } %v49, 0
  %v104 = ptrtoint ptr %v103 to i64
  %v105 = sub i64 %v104, 0
  %v106 = icmp ule i64 %v105, 0
  %v107 = add i64 %v105, 0
  %v108 = select i1 %v106, i64 %v107, i64 1
  %v109 = icmp eq i64 %v108, 1
  br i1 %v109, label %bb23, label %bb22

bb22:                                             ; preds = %bb21
  %v110 = icmp eq i64 %v108, 0
  br i1 %v110, label %bb24, label %bb5

bb23:                                             ; preds = %bb21
  %v111 = extractvalue { ptr } %v49, 0
  %v112 = fmul contract float %v37, 2.550000e+02
  %v113 = fmul contract float %v38, 2.550000e+02
  %v114 = fmul contract float %v39, 2.550000e+02
  %v115 = getelementptr inbounds [4 x float], ptr %v14, i32 0, i64 0
  store float %v112, ptr %v115, align 4
  %v116 = getelementptr inbounds [4 x float], ptr %v14, i32 0, i64 1
  store float %v113, ptr %v116, align 4
  %v117 = getelementptr inbounds [4 x float], ptr %v14, i32 0, i64 2
  store float %v114, ptr %v117, align 4
  %v118 = getelementptr inbounds [4 x float], ptr %v14, i32 0, i64 3
  store float 2.550000e+02, ptr %v118, align 4
  %v119 = load [4 x float], ptr %v14, align 4
  %v120 = insertvalue { [4 x float] } undef, [4 x float] %v119, 0
  store { [4 x float] } %v120, ptr %v111, align 16
  br label %bb25

bb24:                                             ; preds = %bb22
  br label %bb25

bb25:                                             ; preds = %bb24, %bb23
  br label %bb27

bb26:                                             ; preds = %bb2
  br label %bb27

bb27:                                             ; preds = %bb26, %bb25
  ret void

bb28:                                             ; preds = %bb4
  %v121 = add i64 %v41, 1
  %v122 = insertvalue { i64, i64 } undef, i64 1, 0
  %v123 = insertvalue { i64, i64 } %v122, i64 %v41, 1
  %v124 = extractvalue { i64, i64 } %v123, 0
  %v125 = extractvalue { i64, i64 } %v123, 1
  br label %bb30

bb29:                                             ; preds = %bb4
  %v126 = insertvalue { i64, i64 } undef, i64 0, 0
  %v127 = extractvalue { i64, i64 } %v126, 0
  %v128 = extractvalue { i64, i64 } %v126, 1
  br label %bb30

bb30:                                             ; preds = %bb29, %bb28
  %v129 = phi i64 [ %v121, %bb28 ], [ %v41, %bb29 ]
  %v130 = phi i64 [ %v124, %bb28 ], [ %v127, %bb29 ]
  %v131 = phi i64 [ %v125, %bb28 ], [ %v128, %bb29 ]
  %v132 = insertvalue { i64, i64 } undef, i64 %v130, 0
  %v133 = insertvalue { i64, i64 } %v132, i64 %v131, 1
  %v134 = extractvalue { i64, i64 } %v133, 0
  %v135 = bitcast i64 %v134 to i64
  %v136 = icmp eq i64 %v135, 0
  br i1 %v136, label %bb7, label %bb31

bb31:                                             ; preds = %bb30
  %v137 = icmp eq i64 %v135, 1
  br i1 %v137, label %bb6, label %bb5

bb32:                                             ; preds = %bb11
  %v138 = call float @__nv_sqrtf(float %v68) #0
  br label %bb33

bb33:                                             ; preds = %bb32
  %v139 = fdiv contract float %v73, %v138
  %v140 = add i64 %v45, 6
  %v141 = icmp ult i64 %v140, %v47
  br i1 %v141, label %bb12, label %bb40

bb34:                                             ; preds = %bb6
  call void @llvm.trap() #0
  unreachable

bb35:                                             ; preds = %bb8
  call void @llvm.trap() #0
  unreachable

bb36:                                             ; preds = %bb9
  call void @llvm.trap() #0
  unreachable

bb37:                                             ; preds = %bb15
  call void @llvm.trap() #0
  unreachable

bb38:                                             ; preds = %bb16
  call void @llvm.trap() #0
  unreachable

bb39:                                             ; preds = %bb17
  call void @llvm.trap() #0
  unreachable

bb40:                                             ; preds = %bb33
  call void @llvm.trap() #0
  unreachable
}

; Function Attrs: alwaysinline convergent
define { i64, i64 } @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_(ptr %v0, ptr %v1) #1 {
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

; Function Attrs: convergent
define { ptr } @_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCs6HixiQUBNAA_9raytracer4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_(ptr %v0, i64 %v1) #0 {
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

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: alwaysinline nounwind
define internal float @__nv_sqrtf(float %x) #4 {
  %1 = call i32 @__nvvm_reflect(ptr @.str) #7
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %10

3:                                                ; preds = %0
  %4 = call i32 @__nvvm_reflect(ptr @.str.2) #7
  %5 = icmp ne i32 %4, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %3
  %7 = call float @llvm.nvvm.sqrt.rn.ftz.f(float %x) #7
  br label %__nvvm_sqrt_f.exit

8:                                                ; preds = %3
  %9 = call float @llvm.nvvm.sqrt.approx.ftz.f(float %x) #7
  br label %__nvvm_sqrt_f.exit

10:                                               ; preds = %0
  %11 = call i32 @__nvvm_reflect(ptr @.str.2) #7
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %15

13:                                               ; preds = %10
  %14 = call float @llvm.nvvm.sqrt.rn.f(float %x) #7
  br label %__nvvm_sqrt_f.exit

15:                                               ; preds = %10
  %16 = call float @llvm.nvvm.sqrt.approx.f(float %x) #7
  br label %__nvvm_sqrt_f.exit

__nvvm_sqrt_f.exit:                               ; preds = %6, %8, %13, %15
  %.0 = phi float [ %7, %6 ], [ %9, %8 ], [ %14, %13 ], [ %16, %15 ]
  ret float %.0
}

declare i32 @__nvvm_reflect(ptr) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #6

attributes #0 = { convergent }
attributes #1 = { alwaysinline convergent }
attributes #2 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { alwaysinline nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #7 = { nounwind }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
