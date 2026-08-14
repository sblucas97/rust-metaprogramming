; ModuleID = 'builtin.module'
source_filename = "ripple"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare float @__nv_sqrtf(float)
declare float @__nv_cosf(float)
declare float @llvm.floor.f32(float)

define ptx_kernel void @ripple(ptr %v0, i64 %v1, i32 %v2, float %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64, i32 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64, i32 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64, i32 } %v5, i32 %v2, 2
  br label %bb0
bb0:
  %v7 = phi { ptr, i64, i32 } [ %v6, %entry ]
  %v8 = phi float [ %v3, %entry ]
  %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_ = alloca { ptr, i64, i32 }, align 8
  %v10 = alloca {  }, align 1
  %v11 = alloca [4 x float], align 4
  store { ptr, i64, i32 } %v7, ptr %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_, align 8
  %v12 = bitcast ptr %v10 to ptr
  %v13 = bitcast ptr %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_ to ptr
  %v14 = call { i64, i64 } @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_(ptr %v12, ptr %v13) #0
  br label %bb1
bb1:
  %v15 = extractvalue { i64, i64 } %v14, 0
  %v16 = bitcast i64 %v15 to i64
  %v17 = icmp eq i64 %v16, 1
  br i1 %v17, label %bb3, label %bb2
bb2:
  %v18 = icmp eq i64 %v16, 0
  br i1 %v18, label %bb9, label %bb14
bb3:
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
  %v30 = fmul contract float 0.5, %v29
  %v31 = fdiv contract float %v28, 15.0
  %v32 = fsub contract float %v30, %v31
  %v33 = uitofp i32 %v25 to float
  %v34 = fmul contract float 0.5, %v33
  %v35 = fsub contract float %v34, %v31
  %v36 = fmul contract float %v32, %v32
  %v37 = fmul contract float %v35, %v35
  %v38 = fadd contract float %v36, %v37
  %v39 = call float @__nv_sqrtf(float %v38) #0
  br label %bb11
bb4:
  %v40 = extractvalue { ptr } %v64, 0
  %v41 = ptrtoint ptr %v40 to i64
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  br i1 %v46, label %bb6, label %bb5
bb5:
  %v47 = icmp eq i64 %v45, 0
  br i1 %v47, label %bb7, label %bb14
bb6:
  %v48 = extractvalue { ptr } %v64, 0
  %v49 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 0
  store float %v63, ptr %v49, align 4
  %v50 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 1
  store float %v63, ptr %v50, align 4
  %v51 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 2
  store float %v63, ptr %v51, align 4
  %v52 = getelementptr inbounds [4 x float], ptr %v11, i32 0, i64 3
  store float 255.0, ptr %v52, align 4
  %v53 = load [4 x float], ptr %v11, align 4
  %v54 = insertvalue { [4 x float] } undef, [4 x float] %v53, 0
  store { [4 x float] } %v54, ptr %v48, align 16
  br label %bb8
bb7:
  br label %bb8
bb8:
  br label %bb10
bb9:
  br label %bb10
bb10:
  ret void
bb11:
  %v55 = fdiv contract float %v39, 10.0
  %v56 = fdiv contract float %v8, 7.0
  %v57 = fsub contract float %v55, %v56
  %v58 = call float @__nv_cosf(float %v57) #0
  br label %bb12
bb12:
  %v59 = fmul contract float 127.0, %v58
  %v60 = fadd contract float %v55, 1.0
  %v61 = fdiv contract float %v59, %v60
  %v62 = fadd contract float 128.0, %v61
  %v63 = call float @llvm.floor.f32(float %v62) #0
  br label %bb13
bb13:
  %v64 = call { ptr } @_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCslj41F3F2J0U_6ripple4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_(ptr %__cuda_oxide_local_x31093234096f7574094469736a6f696e74536c696365_, i64 %v19) #0
  br label %bb4
bb14:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define { i64, i64 } @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCslj41F3F2J0U_6ripple4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_(ptr %v0, ptr %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load { ptr, i64, i32 }, ptr %v3, align 8
  %v5 = extractvalue { ptr, i64, i32 } %v4, 2
  %v6 = zext i32 %v5 to i64
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb1, label %bb2
bb1:
  %v8 = insertvalue { i64, i64 } undef, i64 0, 0
  %v9 = extractvalue { i64, i64 } %v8, 0
  %v10 = extractvalue { i64, i64 } %v8, 1
  br label %bb21
bb2:
  %v11 = icmp eq i8 0, 1
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb22, label %bb23
bb3:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb5
bb4:
  %v14 = insertvalue { i64, i64 } undef, i64 0, 0
  %v15 = extractvalue { i64, i64 } %v14, 0
  %v16 = extractvalue { i64, i64 } %v14, 1
  br label %bb21
bb5:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb6
bb6:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb7
bb7:
  %v19 = zext i32 %v13 to i64
  %v20 = zext i32 %v17 to i64
  %v21 = zext i32 %v18 to i64
  %v22 = icmp eq i64 %v20, 0
  br i1 %v22, label %bb32, label %bb30
bb8:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb9
bb9:
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb10
bb10:
  %v25 = zext i32 %v70 to i64
  %v26 = zext i32 %v23 to i64
  %v27 = zext i32 %v24 to i64
  %v28 = icmp eq i64 %v26, 0
  br i1 %v28, label %bb37, label %bb35
bb11:
  br label %bb16
bb12:
  %v29 = icmp eq i64 %v77, 18446744073709551615
  br i1 %v29, label %bb13, label %bb14
bb13:
  br label %bb16
bb14:
  %v30 = icmp uge i64 %v77, %v6
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb17, label %bb15
bb15:
  br label %bb16
bb16:
  %v32 = insertvalue { i64, i64 } undef, i64 0, 0
  %v33 = extractvalue { i64, i64 } %v32, 0
  %v34 = extractvalue { i64, i64 } %v32, 1
  br label %bb20
bb17:
  %v35 = icmp ugt i64 %v69, 4294967295
  %v36 = xor i1 %v35, 1
  br i1 %v36, label %bb19, label %bb18
bb18:
  %v37 = insertvalue { i64, i64 } undef, i64 0, 0
  %v38 = extractvalue { i64, i64 } %v37, 0
  %v39 = extractvalue { i64, i64 } %v37, 1
  br label %bb20
bb19:
  %v40 = zext i32 32 to i64
  %v41 = and i64 %v40, 63
  %v42 = shl i64 %v69, %v41
  %v43 = or i64 %v42, %v77
  %v44 = icmp eq i64 %v43, 18446744073709551615
  br i1 %v44, label %bb40, label %bb41
bb20:
  %v45 = phi i64 [ %v33, %bb16 ], [ %v38, %bb18 ]
  %v46 = phi i64 [ %v34, %bb16 ], [ %v39, %bb18 ]
  %v47 = insertvalue { i64, i64 } undef, i64 %v45, 0
  %v48 = insertvalue { i64, i64 } %v47, i64 %v46, 1
  %v49 = extractvalue { i64, i64 } %v48, 0
  %v50 = extractvalue { i64, i64 } %v48, 1
  br label %bb21
bb21:
  %v51 = phi i64 [ %v9, %bb1 ], [ %v15, %bb4 ], [ %v49, %bb20 ], [ %v82, %bb41 ]
  %v52 = phi i64 [ %v10, %bb1 ], [ %v16, %bb4 ], [ %v50, %bb20 ], [ %v83, %bb41 ]
  %v53 = insertvalue { i64, i64 } undef, i64 %v51, 0
  %v54 = insertvalue { i64, i64 } %v53, i64 %v52, 1
  ret { i64, i64 } %v54
bb22:
  %v55 = icmp eq i8 0, 2
  %v56 = xor i1 %v55, 1
  br i1 %v56, label %bb24, label %bb23
bb23:
  br label %bb29
bb24:
  %v57 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb25
bb25:
  %v58 = icmp eq i32 %v57, 1
  br i1 %v58, label %bb26, label %bb27
bb26:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb28
bb27:
  br label %bb29
bb28:
  %v60 = icmp eq i32 %v59, 1
  br label %bb29
bb29:
  %v61 = phi i1 [ 1, %bb23 ], [ 0, %bb27 ], [ %v60, %bb28 ]
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb4, label %bb3
bb30:
  %v63 = sub i64 18446744073709551615, %v21
  %v64 = udiv i64 %v63, %v20
  %v65 = icmp ugt i64 %v19, %v64
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb33, label %bb31
bb31:
  br label %bb32
bb32:
  br label %bb34
bb33:
  %v67 = mul i64 %v19, %v20
  %v68 = add i64 %v67, %v21
  br label %bb34
bb34:
  %v69 = phi i64 [ 18446744073709551615, %bb32 ], [ %v68, %bb33 ]
  %v70 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb8
bb35:
  %v71 = sub i64 18446744073709551615, %v27
  %v72 = udiv i64 %v71, %v26
  %v73 = icmp ugt i64 %v25, %v72
  %v74 = xor i1 %v73, 1
  br i1 %v74, label %bb38, label %bb36
bb36:
  br label %bb37
bb37:
  br label %bb39
bb38:
  %v75 = mul i64 %v25, %v26
  %v76 = add i64 %v75, %v27
  br label %bb39
bb39:
  %v77 = phi i64 [ 18446744073709551615, %bb37 ], [ %v76, %bb38 ]
  %v78 = icmp eq i64 %v69, 18446744073709551615
  br i1 %v78, label %bb11, label %bb12
bb40:
  br label %bb41
bb41:
  %v79 = phi i64 [ %v43, %bb19 ], [ 18446744073709551615, %bb40 ]
  %v80 = insertvalue { i64, i64 } undef, i64 1, 0
  %v81 = insertvalue { i64, i64 } %v80, i64 %v79, 1
  %v82 = extractvalue { i64, i64 } %v81, 0
  %v83 = extractvalue { i64, i64 } %v81, 1
  br label %bb21
}

define { ptr } @_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCslj41F3F2J0U_6ripple4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_(ptr %v0, i64 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = icmp eq i64 16, 0
  %v5 = xor i1 %v4, 1
  br i1 %v5, label %bb1, label %bb3
bb1:
  %v6 = icmp eq i64 %v3, 18446744073709551615
  br i1 %v6, label %bb3, label %bb2
bb2:
  %v7 = zext i32 32 to i64
  %v8 = and i64 %v7, 63
  %v9 = lshr i64 %v3, %v8
  %v10 = and i64 %v3, 4294967295
  %v11 = load { ptr, i64, i32 }, ptr %v2, align 8
  %v12 = extractvalue { ptr, i64, i32 } %v11, 2
  %v13 = zext i32 %v12 to i64
  %v14 = icmp uge i64 %v10, %v13
  %v15 = xor i1 %v14, 1
  br i1 %v15, label %bb5, label %bb4
bb3:
  %v16 = inttoptr i64 0 to ptr
  %v17 = insertvalue { ptr } undef, ptr %v16, 0
  %v18 = extractvalue { ptr } %v17, 0
  br label %bb9
bb4:
  %v19 = inttoptr i64 0 to ptr
  %v20 = insertvalue { ptr } undef, ptr %v19, 0
  %v21 = extractvalue { ptr } %v20, 0
  br label %bb9
bb5:
  %v22 = mul i64 %v9, %v13
  %v23 = add i64 %v22, %v10
  %v24 = load { ptr, i64, i32 }, ptr %v2, align 8
  %v25 = extractvalue { ptr, i64, i32 } %v24, 1
  %v26 = icmp ult i64 %v23, %v25
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb7, label %bb6
bb6:
  %v28 = load { ptr, i64, i32 }, ptr %v2, align 8
  %v29 = extractvalue { ptr, i64, i32 } %v28, 0
  %v30 = getelementptr inbounds { [4 x float] }, ptr %v29, i64 %v23
  %v31 = insertvalue { ptr } undef, ptr %v30, 0
  %v32 = extractvalue { ptr } %v31, 0
  br label %bb8
bb7:
  %v33 = inttoptr i64 0 to ptr
  %v34 = insertvalue { ptr } undef, ptr %v33, 0
  %v35 = extractvalue { ptr } %v34, 0
  br label %bb8
bb8:
  %v36 = phi ptr [ %v32, %bb6 ], [ %v35, %bb7 ]
  %v37 = insertvalue { ptr } undef, ptr %v36, 0
  %v38 = extractvalue { ptr } %v37, 0
  br label %bb9
bb9:
  %v39 = phi ptr [ %v18, %bb3 ], [ %v21, %bb4 ], [ %v38, %bb8 ]
  %v40 = insertvalue { ptr } undef, ptr %v39, 0
  ret { ptr } %v40
}


@llvm.used = appending global [1 x ptr] [ptr @ripple], section "llvm.metadata"

attributes #0 = { convergent }
