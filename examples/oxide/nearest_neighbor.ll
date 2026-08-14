; ModuleID = 'builtin.module'
source_filename = "nearest_neighbor"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.trap()
declare float @__nv_sqrtf(float)

define ptx_kernel void @euclid(ptr %v0, i64 %v1, ptr %v2, i64 %v3, float %v4, float %v5) #0 {
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
  %__cuda_oxide_local_x320931360964697374616e636573094469736a6f696e74536c696365_ = alloca { ptr, i64 }, align 8
  %v15 = alloca {  }, align 1
  store { ptr, i64 } %v11, ptr %__cuda_oxide_local_x320931360964697374616e636573094469736a6f696e74536c696365_, align 8
  %v16 = bitcast ptr %v15 to ptr
  %v17 = call { ptr, i64 } @_RINvMsu_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB6_13DisjointSlicefE15get_mut_indexedNtNtNtB8_6thread10___internal13UnknownDomainNtB1o_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %__cuda_oxide_local_x320931360964697374616e636573094469736a6f696e74536c696365_, ptr %v16) #0
  br label %bb1
bb1:
  %v18 = extractvalue { ptr, i64 } %v17, 0
  %v19 = ptrtoint ptr %v18 to i64
  %v20 = sub i64 %v19, 0
  %v21 = icmp ule i64 %v20, 0
  %v22 = add i64 %v20, 0
  %v23 = select i1 %v21, i64 %v22, i64 1
  %v24 = icmp eq i64 %v23, 1
  br i1 %v24, label %bb3, label %bb2
bb2:
  %v25 = icmp eq i64 %v23, 0
  br i1 %v25, label %bb6, label %bb9
bb3:
  %v26 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v17, ptr %v26, align 8
  %v27 = getelementptr inbounds i8, ptr %v26, i64 0
  %v28 = load { ptr, i64 }, ptr %v27, align 8
  %v29 = extractvalue { ptr, i64 } %v28, 0
  %v30 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v17, ptr %v30, align 8
  %v31 = getelementptr inbounds i8, ptr %v30, i64 0
  %v32 = load { ptr, i64 }, ptr %v31, align 8
  %v33 = extractvalue { ptr, i64 } %v32, 1
  %v34 = mul i64 2, %v33
  %v35 = extractvalue { ptr, i64 } %v10, 1
  %v36 = icmp ult i64 %v34, %v35
  br i1 %v36, label %bb4, label %bb10
bb4:
  %v37 = extractvalue { ptr, i64 } %v10, 0
  %v38 = getelementptr inbounds float, ptr %v37, i64 %v34
  %v39 = load float, ptr %v38, align 4
  %v40 = fsub contract float %v12, %v39
  %v41 = add i64 %v34, 1
  %v42 = icmp ult i64 %v41, %v35
  br i1 %v42, label %bb5, label %bb11
bb5:
  %v43 = extractvalue { ptr, i64 } %v10, 0
  %v44 = getelementptr inbounds float, ptr %v43, i64 %v41
  %v45 = load float, ptr %v44, align 4
  %v46 = fsub contract float %v13, %v45
  %v47 = fmul contract float %v40, %v40
  %v48 = fmul contract float %v46, %v46
  %v49 = fadd contract float %v47, %v48
  %v50 = call float @__nv_sqrtf(float %v49) #0
  br label %bb8
bb6:
  br label %bb7
bb7:
  ret void
bb8:
  store float %v50, ptr %v29, align 4
  br label %bb7
bb9:
  unreachable
bb10:
  call void @llvm.trap() #0
  unreachable
bb11:
  call void @llvm.trap() #0
  unreachable
}

define { ptr, i64 } @_RINvMsu_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB6_13DisjointSlicefE15get_mut_indexedNtNtNtB8_6thread10___internal13UnknownDomainNtB1o_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0, ptr %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = call { i64, i64 } @_RINvXs4_NtCsiWbdGYq5HZh_11cuda_device6threadNtB6_7Index1DNtB6_12IndexFormula10from_scopeNtNtB6_10___internal13UnknownDomainNtB1q_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v3) #0
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
  %v10 = icmp ne i64 4, 0
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
  %v23 = getelementptr inbounds float, ptr %v22, i64 %v9
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

define { i64, i64 } @_RINvXs4_NtCsiWbdGYq5HZh_11cuda_device6threadNtB6_7Index1DNtB6_12IndexFormula10from_scopeNtNtB6_10___internal13UnknownDomainNtB1q_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i64 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v1) #0
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

define i64 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0) alwaysinline #0 {
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
  %v20 = call i1 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v1) #0
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

define i1 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0) alwaysinline #0 {
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


@llvm.used = appending global [1 x ptr] [ptr @euclid], section "llvm.metadata"

attributes #0 = { convergent }
