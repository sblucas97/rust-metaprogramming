; ModuleID = 'builtin.module'
source_filename = "nbodies"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.trap()

define ptx_kernel void @gpu_integrate(ptr %v0, i64 %v1, ptr %v2, i64 %v3, float %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi float [ %v4, %entry ]
  %__cuda_oxide_local_x3109313609706f73094469736a6f696e74536c696365_ = alloca { ptr, i64 }, align 8
  %v13 = alloca {  }, align 1
  %v14 = alloca [3 x float], align 4
  store { ptr, i64 } %v9, ptr %__cuda_oxide_local_x3109313609706f73094469736a6f696e74536c696365_, align 8
  %v15 = bitcast ptr %v13 to ptr
  %v16 = call { ptr, i64 } @_RINvMsu_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB6_13DisjointSliceNtCs2nOxImCifRx_7nbodies4Vec3E15get_mut_indexedNtNtNtB8_6thread10___internal13UnknownDomainNtB1Q_17NativeCoordinatesEB15_(ptr %__cuda_oxide_local_x3109313609706f73094469736a6f696e74536c696365_, ptr %v15) #0
  br label %bb1
bb1:
  %v17 = extractvalue { ptr, i64 } %v16, 0
  %v18 = ptrtoint ptr %v17 to i64
  %v19 = sub i64 %v18, 0
  %v20 = icmp ule i64 %v19, 0
  %v21 = add i64 %v19, 0
  %v22 = select i1 %v20, i64 %v21, i64 1
  %v23 = icmp eq i64 %v22, 1
  br i1 %v23, label %bb3, label %bb2
bb2:
  %v24 = icmp eq i64 %v22, 0
  br i1 %v24, label %bb5, label %bb7
bb3:
  %v25 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v16, ptr %v25, align 8
  %v26 = getelementptr inbounds i8, ptr %v25, i64 0
  %v27 = load { ptr, i64 }, ptr %v26, align 8
  %v28 = extractvalue { ptr, i64 } %v27, 0
  %v29 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v16, ptr %v29, align 8
  %v30 = getelementptr inbounds i8, ptr %v29, i64 0
  %v31 = load { ptr, i64 }, ptr %v30, align 8
  %v32 = extractvalue { ptr, i64 } %v31, 1
  %v33 = extractvalue { ptr, i64 } %v10, 1
  %v34 = icmp ult i64 %v32, %v33
  br i1 %v34, label %bb4, label %bb8
bb4:
  %v35 = extractvalue { ptr, i64 } %v10, 0
  %v36 = getelementptr inbounds { [3 x float] }, ptr %v35, i64 %v32
  %v37 = getelementptr inbounds { [3 x float] }, ptr %v36, i32 0, i32 0
  %v38 = load [3 x float], ptr %v37, align 4
  store [3 x float] %v38, ptr %v14, align 4
  %v39 = getelementptr inbounds { [3 x float] }, ptr %v28, i32 0, i32 0
  %v40 = getelementptr inbounds [3 x float], ptr %v39, i32 0, i64 0
  %v41 = load float, ptr %v40, align 4
  %v42 = getelementptr inbounds [3 x float], ptr %v14, i32 0, i64 0
  %v43 = load float, ptr %v42, align 4
  %v44 = fmul contract float %v43, %v11
  %v45 = fadd contract float %v41, %v44
  %v46 = getelementptr inbounds { [3 x float] }, ptr %v28, i32 0, i32 0
  %v47 = getelementptr inbounds [3 x float], ptr %v46, i32 0, i64 0
  store float %v45, ptr %v47, align 4
  %v48 = getelementptr inbounds { [3 x float] }, ptr %v28, i32 0, i32 0
  %v49 = getelementptr inbounds [3 x float], ptr %v48, i32 0, i64 1
  %v50 = load float, ptr %v49, align 4
  %v51 = getelementptr inbounds [3 x float], ptr %v14, i32 0, i64 1
  %v52 = load float, ptr %v51, align 4
  %v53 = fmul contract float %v52, %v11
  %v54 = fadd contract float %v50, %v53
  %v55 = getelementptr inbounds { [3 x float] }, ptr %v28, i32 0, i32 0
  %v56 = getelementptr inbounds [3 x float], ptr %v55, i32 0, i64 1
  store float %v54, ptr %v56, align 4
  %v57 = getelementptr inbounds { [3 x float] }, ptr %v28, i32 0, i32 0
  %v58 = getelementptr inbounds [3 x float], ptr %v57, i32 0, i64 2
  %v59 = load float, ptr %v58, align 4
  %v60 = getelementptr inbounds [3 x float], ptr %v14, i32 0, i64 2
  %v61 = load float, ptr %v60, align 4
  %v62 = fmul contract float %v61, %v11
  %v63 = fadd contract float %v59, %v62
  %v64 = getelementptr inbounds { [3 x float] }, ptr %v28, i32 0, i32 0
  %v65 = getelementptr inbounds [3 x float], ptr %v64, i32 0, i64 2
  store float %v63, ptr %v65, align 4
  br label %bb6
bb5:
  br label %bb6
bb6:
  ret void
bb7:
  unreachable
bb8:
  call void @llvm.trap() #0
  unreachable
}

declare float @__nv_sqrtf(float)

define ptx_kernel void @gpu_n_bodies(ptr %v0, i64 %v1, ptr %v2, i64 %v3, float %v4, float %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi { ptr, i64 } [ %v9, %entry ]
  %v12 = phi float [ %v4, %entry ]
  %v13 = phi float [ %v5, %entry ]
  %__cuda_oxide_local_x320931360976656c094469736a6f696e74536c696365_ = alloca { ptr, i64 }, align 8
  %v15 = alloca {  }, align 1
  %v16 = alloca [3 x float], align 4
  %v17 = alloca [3 x float], align 4
  store { ptr, i64 } %v11, ptr %__cuda_oxide_local_x320931360976656c094469736a6f696e74536c696365_, align 8
  %v18 = bitcast ptr %v15 to ptr
  %v19 = call { ptr, i64 } @_RINvMsu_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB6_13DisjointSliceNtCs2nOxImCifRx_7nbodies4Vec3E15get_mut_indexedNtNtNtB8_6thread10___internal13UnknownDomainNtB1Q_17NativeCoordinatesEB15_(ptr %__cuda_oxide_local_x320931360976656c094469736a6f696e74536c696365_, ptr %v18) #0
  br label %bb1
bb1:
  %v20 = extractvalue { ptr, i64 } %v19, 0
  %v21 = ptrtoint ptr %v20 to i64
  %v22 = sub i64 %v21, 0
  %v23 = icmp ule i64 %v22, 0
  %v24 = add i64 %v22, 0
  %v25 = select i1 %v23, i64 %v24, i64 1
  %v26 = icmp eq i64 %v25, 1
  br i1 %v26, label %bb3, label %bb2
bb2:
  %v27 = icmp eq i64 %v25, 0
  br i1 %v27, label %bb10, label %bb6
bb3:
  %v28 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v19, ptr %v28, align 8
  %v29 = getelementptr inbounds i8, ptr %v28, i64 0
  %v30 = load { ptr, i64 }, ptr %v29, align 8
  %v31 = extractvalue { ptr, i64 } %v30, 0
  %v32 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v19, ptr %v32, align 8
  %v33 = getelementptr inbounds i8, ptr %v32, i64 0
  %v34 = load { ptr, i64 }, ptr %v33, align 8
  %v35 = extractvalue { ptr, i64 } %v34, 1
  %v36 = extractvalue { ptr, i64 } %v10, 1
  %v37 = icmp ult i64 %v35, %v36
  br i1 %v37, label %bb4, label %bb17
bb4:
  %v38 = extractvalue { ptr, i64 } %v10, 0
  %v39 = getelementptr inbounds { [3 x float] }, ptr %v38, i64 %v35
  %v40 = getelementptr inbounds { [3 x float] }, ptr %v39, i32 0, i32 0
  %v41 = load [3 x float], ptr %v40, align 4
  store [3 x float] %v41, ptr %v16, align 4
  br label %bb5
bb5:
  %v42 = phi float [ 0.0, %bb4 ], [ %v118, %bb16 ]
  %v43 = phi float [ 0.0, %bb4 ], [ %v120, %bb16 ]
  %v44 = phi float [ 0.0, %bb4 ], [ %v122, %bb16 ]
  %v45 = phi i64 [ 0, %bb4 ], [ %v105, %bb16 ]
  %v46 = icmp ult i64 %v45, %v36
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb13, label %bb12
bb6:
  unreachable
bb7:
  %v48 = extractvalue { i64, i64 } %v109, 1
  %v49 = icmp ult i64 %v48, %v36
  br i1 %v49, label %bb9, label %bb18
bb8:
  %v50 = getelementptr inbounds { [3 x float] }, ptr %v31, i32 0, i32 0
  %v51 = getelementptr inbounds [3 x float], ptr %v50, i32 0, i64 0
  %v52 = load float, ptr %v51, align 4
  %v53 = fmul contract float %v12, %v42
  %v54 = fadd contract float %v52, %v53
  %v55 = getelementptr inbounds { [3 x float] }, ptr %v31, i32 0, i32 0
  %v56 = getelementptr inbounds [3 x float], ptr %v55, i32 0, i64 0
  store float %v54, ptr %v56, align 4
  %v57 = getelementptr inbounds { [3 x float] }, ptr %v31, i32 0, i32 0
  %v58 = getelementptr inbounds [3 x float], ptr %v57, i32 0, i64 1
  %v59 = load float, ptr %v58, align 4
  %v60 = fmul contract float %v12, %v43
  %v61 = fadd contract float %v59, %v60
  %v62 = getelementptr inbounds { [3 x float] }, ptr %v31, i32 0, i32 0
  %v63 = getelementptr inbounds [3 x float], ptr %v62, i32 0, i64 1
  store float %v61, ptr %v63, align 4
  %v64 = getelementptr inbounds { [3 x float] }, ptr %v31, i32 0, i32 0
  %v65 = getelementptr inbounds [3 x float], ptr %v64, i32 0, i64 2
  %v66 = load float, ptr %v65, align 4
  %v67 = fmul contract float %v12, %v44
  %v68 = fadd contract float %v66, %v67
  %v69 = getelementptr inbounds { [3 x float] }, ptr %v31, i32 0, i32 0
  %v70 = getelementptr inbounds [3 x float], ptr %v69, i32 0, i64 2
  store float %v68, ptr %v70, align 4
  br label %bb11
bb9:
  %v71 = extractvalue { ptr, i64 } %v10, 0
  %v72 = getelementptr inbounds { [3 x float] }, ptr %v71, i64 %v48
  %v73 = getelementptr inbounds { [3 x float] }, ptr %v72, i32 0, i32 0
  %v74 = load [3 x float], ptr %v73, align 4
  store [3 x float] %v74, ptr %v17, align 4
  %v75 = getelementptr inbounds [3 x float], ptr %v17, i32 0, i64 0
  %v76 = load float, ptr %v75, align 4
  %v77 = getelementptr inbounds [3 x float], ptr %v16, i32 0, i64 0
  %v78 = load float, ptr %v77, align 4
  %v79 = fsub contract float %v76, %v78
  %v80 = getelementptr inbounds [3 x float], ptr %v17, i32 0, i64 1
  %v81 = load float, ptr %v80, align 4
  %v82 = getelementptr inbounds [3 x float], ptr %v16, i32 0, i64 1
  %v83 = load float, ptr %v82, align 4
  %v84 = fsub contract float %v81, %v83
  %v85 = getelementptr inbounds [3 x float], ptr %v17, i32 0, i64 2
  %v86 = load float, ptr %v85, align 4
  %v87 = getelementptr inbounds [3 x float], ptr %v16, i32 0, i64 2
  %v88 = load float, ptr %v87, align 4
  %v89 = fsub contract float %v86, %v88
  %v90 = fmul contract float %v79, %v79
  %v91 = fmul contract float %v84, %v84
  %v92 = fadd contract float %v90, %v91
  %v93 = fmul contract float %v89, %v89
  %v94 = fadd contract float %v92, %v93
  %v95 = fadd contract float %v94, %v13
  %v96 = call float @__nv_sqrtf(float %v95) #0
  br label %bb16
bb10:
  br label %bb11
bb11:
  ret void
bb12:
  %v97 = add i64 %v45, 1
  %v98 = insertvalue { i64, i64 } undef, i64 1, 0
  %v99 = insertvalue { i64, i64 } %v98, i64 %v45, 1
  %v100 = extractvalue { i64, i64 } %v99, 0
  %v101 = extractvalue { i64, i64 } %v99, 1
  br label %bb14
bb13:
  %v102 = insertvalue { i64, i64 } undef, i64 0, 0
  %v103 = extractvalue { i64, i64 } %v102, 0
  %v104 = extractvalue { i64, i64 } %v102, 1
  br label %bb14
bb14:
  %v105 = phi i64 [ %v97, %bb12 ], [ %v45, %bb13 ]
  %v106 = phi i64 [ %v100, %bb12 ], [ %v103, %bb13 ]
  %v107 = phi i64 [ %v101, %bb12 ], [ %v104, %bb13 ]
  %v108 = insertvalue { i64, i64 } undef, i64 %v106, 0
  %v109 = insertvalue { i64, i64 } %v108, i64 %v107, 1
  %v110 = extractvalue { i64, i64 } %v109, 0
  %v111 = bitcast i64 %v110 to i64
  %v112 = icmp eq i64 %v111, 0
  br i1 %v112, label %bb8, label %bb15
bb15:
  %v113 = icmp eq i64 %v111, 1
  br i1 %v113, label %bb7, label %bb6
bb16:
  %v114 = fdiv contract float 1.0, %v96
  %v115 = fmul contract float %v114, %v114
  %v116 = fmul contract float %v115, %v114
  %v117 = fmul contract float %v79, %v116
  %v118 = fadd contract float %v42, %v117
  %v119 = fmul contract float %v84, %v116
  %v120 = fadd contract float %v43, %v119
  %v121 = fmul contract float %v89, %v116
  %v122 = fadd contract float %v44, %v121
  br label %bb5
bb17:
  call void @llvm.trap() #0
  unreachable
bb18:
  call void @llvm.trap() #0
  unreachable
}

define { ptr, i64 } @_RINvMsu_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB6_13DisjointSliceNtCs2nOxImCifRx_7nbodies4Vec3E15get_mut_indexedNtNtNtB8_6thread10___internal13UnknownDomainNtB1Q_17NativeCoordinatesEB15_(ptr %v0, ptr %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = call { i64, i64 } @_RINvXs4_NtCsiWbdGYq5HZh_11cuda_device6threadNtB6_7Index1DNtB6_12IndexFormula10from_scopeNtNtB6_10___internal13UnknownDomainNtB1q_17NativeCoordinatesECs2nOxImCifRx_7nbodies(ptr %v3) #0
  br label %bb1
bb1:
  %v5 = extractvalue { i64, i64 } %v4, 0
  %v6 = bitcast i64 %v5 to i64
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb14, label %bb2
bb2:
  %v8 = icmp eq i64 %v6, 1
  br i1 %v8, label %bb15, label %bb3
bb3:
  unreachable
bb4:
  %v9 = extractvalue { i64, i64 } %v48, 1
  %v10 = icmp ne i64 12, 0
  %v11 = xor i1 %v10, 1
  br i1 %v11, label %bb9, label %bb6
bb5:
  %v12 = icmp eq i64 0, 0
  %v13 = inttoptr i64 0 to ptr
  %v14 = insertvalue { ptr, i64 } undef, ptr %v13, 0
  %v15 = extractvalue { ptr, i64 } %v14, 0
  %v16 = extractvalue { ptr, i64 } %v14, 1
  br label %bb11
bb6:
  %v17 = load { ptr, i64 }, ptr %v2, align 8
  %v18 = extractvalue { ptr, i64 } %v17, 1
  %v19 = icmp ult i64 %v9, %v18
  %v20 = xor i1 %v19, 1
  br i1 %v20, label %bb8, label %bb7
bb7:
  %v21 = load { ptr, i64 }, ptr %v2, align 8
  %v22 = extractvalue { ptr, i64 } %v21, 0
  %v23 = getelementptr inbounds { [3 x float] }, ptr %v22, i64 %v9
  %v24 = insertvalue { ptr, i64 } undef, ptr %v23, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v9, 1
  %v26 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } undef, ptr %v26, align 8
  %v27 = getelementptr inbounds i8, ptr %v26, i64 0
  store { ptr, i64 } %v25, ptr %v27, align 8
  %v28 = load { ptr, i64 }, ptr %v26, align 8
  %v29 = extractvalue { ptr, i64 } %v28, 0
  %v30 = extractvalue { ptr, i64 } %v28, 1
  br label %bb10
bb8:
  br label %bb9
bb9:
  %v31 = inttoptr i64 0 to ptr
  %v32 = insertvalue { ptr, i64 } undef, ptr %v31, 0
  %v33 = extractvalue { ptr, i64 } %v32, 0
  %v34 = extractvalue { ptr, i64 } %v32, 1
  br label %bb10
bb10:
  %v35 = phi ptr [ %v29, %bb7 ], [ %v33, %bb9 ]
  %v36 = phi i64 [ %v30, %bb7 ], [ %v34, %bb9 ]
  %v37 = insertvalue { ptr, i64 } undef, ptr %v35, 0
  %v38 = insertvalue { ptr, i64 } %v37, i64 %v36, 1
  %v39 = extractvalue { ptr, i64 } %v38, 0
  %v40 = extractvalue { ptr, i64 } %v38, 1
  br label %bb11
bb11:
  %v41 = phi ptr [ %v15, %bb5 ], [ %v39, %bb10 ]
  %v42 = phi i64 [ %v16, %bb5 ], [ %v40, %bb10 ]
  %v43 = insertvalue { ptr, i64 } undef, ptr %v41, 0
  %v44 = insertvalue { ptr, i64 } %v43, i64 %v42, 1
  ret { ptr, i64 } %v44
bb12:
  %v45 = phi i64 [ %v54, %bb14 ], [ %v59, %bb15 ]
  %v46 = phi i64 [ %v55, %bb14 ], [ %v60, %bb15 ]
  %v47 = insertvalue { i64, i64 } undef, i64 %v45, 0
  %v48 = insertvalue { i64, i64 } %v47, i64 %v46, 1
  %v49 = extractvalue { i64, i64 } %v48, 0
  %v50 = bitcast i64 %v49 to i64
  %v51 = icmp eq i64 %v50, 0
  br i1 %v51, label %bb4, label %bb13
bb13:
  %v52 = icmp eq i64 %v50, 1
  br i1 %v52, label %bb5, label %bb3
bb14:
  %v53 = insertvalue { i64, i64 } undef, i64 1, 0
  %v54 = extractvalue { i64, i64 } %v53, 0
  %v55 = extractvalue { i64, i64 } %v53, 1
  br label %bb12
bb15:
  %v56 = extractvalue { i64, i64 } %v4, 1
  %v57 = insertvalue { i64, i64 } undef, i64 0, 0
  %v58 = insertvalue { i64, i64 } %v57, i64 %v56, 1
  %v59 = extractvalue { i64, i64 } %v58, 0
  %v60 = extractvalue { i64, i64 } %v58, 1
  br label %bb12
}

define { i64, i64 } @_RINvXs4_NtCsiWbdGYq5HZh_11cuda_device6threadNtB6_7Index1DNtB6_12IndexFormula10from_scopeNtNtB6_10___internal13UnknownDomainNtB1q_17NativeCoordinatesECs2nOxImCifRx_7nbodies(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i64 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs2nOxImCifRx_7nbodies(ptr %v1) #0
  br label %bb1
bb1:
  %v3 = icmp eq i64 %v2, 18446744073709551615
  br i1 %v3, label %bb4, label %bb3
bb2:
  %v4 = phi i64 [ %v10, %bb3 ], [ %v13, %bb4 ]
  %v5 = phi i64 [ %v11, %bb3 ], [ %v14, %bb4 ]
  %v6 = insertvalue { i64, i64 } undef, i64 %v4, 0
  %v7 = insertvalue { i64, i64 } %v6, i64 %v5, 1
  ret { i64, i64 } %v7
bb3:
  %v8 = insertvalue { i64, i64 } undef, i64 1, 0
  %v9 = insertvalue { i64, i64 } %v8, i64 %v2, 1
  %v10 = extractvalue { i64, i64 } %v9, 0
  %v11 = extractvalue { i64, i64 } %v9, 1
  br label %bb2
bb4:
  %v12 = insertvalue { i64, i64 } undef, i64 0, 0
  %v13 = extractvalue { i64, i64 } %v12, 0
  %v14 = extractvalue { i64, i64 } %v12, 1
  br label %bb2
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define i64 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs2nOxImCifRx_7nbodies(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v5 = zext i32 %v2 to i64
  %v6 = zext i32 %v3 to i64
  %v7 = zext i32 %v4 to i64
  %v8 = icmp eq i64 %v6, 0
  br i1 %v8, label %bb10, label %bb8
bb4:
  %v9 = xor i1 %v20, 1
  br i1 %v9, label %bb6, label %bb5
bb5:
  %v10 = icmp ne i64 %v19, 18446744073709551615
  br label %bb7
bb6:
  br label %bb7
bb7:
  %v11 = phi i1 [ %v10, %bb5 ], [ 0, %bb6 ]
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb14, label %bb13
bb8:
  %v13 = sub i64 18446744073709551615, %v7
  %v14 = udiv i64 %v13, %v6
  %v15 = icmp ugt i64 %v5, %v14
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb11, label %bb9
bb9:
  br label %bb10
bb10:
  br label %bb12
bb11:
  %v17 = mul i64 %v5, %v6
  %v18 = add i64 %v17, %v7
  br label %bb12
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs2nOxImCifRx_7nbodies(ptr %v1) #0
  br label %bb4
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v21, label %bb14, label %bb15
bb14:
  br label %bb15
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs2nOxImCifRx_7nbodies(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1
  %v3 = xor i1 %v2, 1
  br i1 %v3, label %bb2, label %bb1
bb1:
  br label %bb8
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb3
bb3:
  %v5 = icmp eq i32 %v4, 1
  br i1 %v5, label %bb4, label %bb5
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  br label %bb6
bb5:
  br label %bb7
bb6:
  %v7 = icmp eq i32 %v6, 1
  br label %bb7
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1
  br i1 %v10, label %bb9, label %bb10
bb9:
  %v11 = icmp eq i8 0, 2
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb12
bb12:
  %v14 = icmp eq i32 %v13, 1
  br i1 %v14, label %bb13, label %bb14
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb15
bb14:
  br label %bb16
bb15:
  %v16 = icmp eq i32 %v15, 1
  br label %bb16
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1
  br i1 %v19, label %bb19, label %bb18
bb18:
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20
}


@llvm.used = appending global [2 x ptr] [ptr @gpu_integrate, ptr @gpu_n_bodies], section "llvm.metadata"

attributes #0 = { convergent }
