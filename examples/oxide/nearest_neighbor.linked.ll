; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @euclid], section "llvm.metadata"
@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1

; Function Attrs: convergent
define ptx_kernel void @euclid(ptr %v0, i64 %v1, ptr %v2, i64 %v3, float %v4, float %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0

bb0:                                              ; preds = %entry
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi { ptr, i64 } [ %v9, %entry ]
  %v12 = phi float [ %v4, %entry ]
  %v13 = phi float [ %v5, %entry ]
  %__cuda_oxide_local_x320931360964697374616e636573094469736a6f696e74536c696365_ = alloca { ptr, i64 }, align 8
  %v15 = alloca {}, align 1
  store { ptr, i64 } %v11, ptr %__cuda_oxide_local_x320931360964697374616e636573094469736a6f696e74536c696365_, align 8
  %v16 = bitcast ptr %v15 to ptr
  %v17 = call { ptr, i64 } @_RINvMsu_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB6_13DisjointSlicefE15get_mut_indexedNtNtNtB8_6thread10___internal13UnknownDomainNtB1o_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %__cuda_oxide_local_x320931360964697374616e636573094469736a6f696e74536c696365_, ptr %v16) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v18 = extractvalue { ptr, i64 } %v17, 0
  %v19 = ptrtoint ptr %v18 to i64
  %v20 = sub i64 %v19, 0
  %v21 = icmp ule i64 %v20, 0
  %v22 = add i64 %v20, 0
  %v23 = select i1 %v21, i64 %v22, i64 1
  %v24 = icmp eq i64 %v23, 1
  br i1 %v24, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  %v25 = icmp eq i64 %v23, 0
  br i1 %v25, label %bb6, label %bb9

bb3:                                              ; preds = %bb1
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

bb4:                                              ; preds = %bb3
  %v37 = extractvalue { ptr, i64 } %v10, 0
  %v38 = getelementptr inbounds float, ptr %v37, i64 %v34
  %v39 = load float, ptr %v38, align 4
  %v40 = fsub contract float %v12, %v39
  %v41 = add i64 %v34, 1
  %v42 = icmp ult i64 %v41, %v35
  br i1 %v42, label %bb5, label %bb11

bb5:                                              ; preds = %bb4
  %v43 = extractvalue { ptr, i64 } %v10, 0
  %v44 = getelementptr inbounds float, ptr %v43, i64 %v41
  %v45 = load float, ptr %v44, align 4
  %v46 = fsub contract float %v13, %v45
  %v47 = fmul contract float %v40, %v40
  %v48 = fmul contract float %v46, %v46
  %v49 = fadd contract float %v47, %v48
  %v50 = call float @__nv_sqrtf(float %v49) #0
  br label %bb8

bb6:                                              ; preds = %bb2
  br label %bb7

bb7:                                              ; preds = %bb8, %bb6
  ret void

bb8:                                              ; preds = %bb5
  store float %v50, ptr %v29, align 4
  br label %bb7

bb9:                                              ; preds = %bb2
  unreachable

bb10:                                             ; preds = %bb3
  call void @llvm.trap() #0
  unreachable

bb11:                                             ; preds = %bb4
  call void @llvm.trap() #0
  unreachable
}

; Function Attrs: convergent
define { ptr, i64 } @_RINvMsu_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB6_13DisjointSlicefE15get_mut_indexedNtNtNtB8_6thread10___internal13UnknownDomainNtB1o_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0, ptr %v1) #0 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = call { i64, i64 } @_RINvXs4_NtCsiWbdGYq5HZh_11cuda_device6threadNtB6_7Index1DNtB6_12IndexFormula10from_scopeNtNtB6_10___internal13UnknownDomainNtB1q_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v3) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v5 = extractvalue { i64, i64 } %v4, 0
  %v6 = bitcast i64 %v5 to i64
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb14, label %bb2

bb2:                                              ; preds = %bb1
  %v8 = icmp eq i64 %v6, 1
  br i1 %v8, label %bb15, label %bb3

bb3:                                              ; preds = %bb13, %bb2
  unreachable

bb4:                                              ; preds = %bb12
  %v9 = extractvalue { i64, i64 } %v48, 1
  %v10 = icmp ne i64 4, 0
  %v11 = xor i1 %v10, true
  br i1 %v11, label %bb9, label %bb6

bb5:                                              ; preds = %bb13
  %v12 = icmp eq i64 0, 0
  %v13 = inttoptr i64 0 to ptr
  %v14 = insertvalue { ptr, i64 } undef, ptr %v13, 0
  %v15 = extractvalue { ptr, i64 } %v14, 0
  %v16 = extractvalue { ptr, i64 } %v14, 1
  br label %bb11

bb6:                                              ; preds = %bb4
  %v17 = load { ptr, i64 }, ptr %v2, align 8
  %v18 = extractvalue { ptr, i64 } %v17, 1
  %v19 = icmp ult i64 %v9, %v18
  %v20 = xor i1 %v19, true
  br i1 %v20, label %bb8, label %bb7

bb7:                                              ; preds = %bb6
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

bb8:                                              ; preds = %bb6
  br label %bb9

bb9:                                              ; preds = %bb8, %bb4
  %v31 = inttoptr i64 0 to ptr
  %v32 = insertvalue { ptr, i64 } undef, ptr %v31, 0
  %v33 = extractvalue { ptr, i64 } %v32, 0
  %v34 = extractvalue { ptr, i64 } %v32, 1
  br label %bb10

bb10:                                             ; preds = %bb9, %bb7
  %v35 = phi ptr [ %v29, %bb7 ], [ %v33, %bb9 ]
  %v36 = phi i64 [ %v30, %bb7 ], [ %v34, %bb9 ]
  %v37 = insertvalue { ptr, i64 } undef, ptr %v35, 0
  %v38 = insertvalue { ptr, i64 } %v37, i64 %v36, 1
  %v39 = extractvalue { ptr, i64 } %v38, 0
  %v40 = extractvalue { ptr, i64 } %v38, 1
  br label %bb11

bb11:                                             ; preds = %bb10, %bb5
  %v41 = phi ptr [ %v15, %bb5 ], [ %v39, %bb10 ]
  %v42 = phi i64 [ %v16, %bb5 ], [ %v40, %bb10 ]
  %v43 = insertvalue { ptr, i64 } undef, ptr %v41, 0
  %v44 = insertvalue { ptr, i64 } %v43, i64 %v42, 1
  ret { ptr, i64 } %v44

bb12:                                             ; preds = %bb15, %bb14
  %v45 = phi i64 [ %v54, %bb14 ], [ %v59, %bb15 ]
  %v46 = phi i64 [ %v55, %bb14 ], [ %v60, %bb15 ]
  %v47 = insertvalue { i64, i64 } undef, i64 %v45, 0
  %v48 = insertvalue { i64, i64 } %v47, i64 %v46, 1
  %v49 = extractvalue { i64, i64 } %v48, 0
  %v50 = bitcast i64 %v49 to i64
  %v51 = icmp eq i64 %v50, 0
  br i1 %v51, label %bb4, label %bb13

bb13:                                             ; preds = %bb12
  %v52 = icmp eq i64 %v50, 1
  br i1 %v52, label %bb5, label %bb3

bb14:                                             ; preds = %bb1
  %v53 = insertvalue { i64, i64 } undef, i64 1, 0
  %v54 = extractvalue { i64, i64 } %v53, 0
  %v55 = extractvalue { i64, i64 } %v53, 1
  br label %bb12

bb15:                                             ; preds = %bb2
  %v56 = extractvalue { i64, i64 } %v4, 1
  %v57 = insertvalue { i64, i64 } undef, i64 0, 0
  %v58 = insertvalue { i64, i64 } %v57, i64 %v56, 1
  %v59 = extractvalue { i64, i64 } %v58, 0
  %v60 = extractvalue { i64, i64 } %v58, 1
  br label %bb12
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: alwaysinline convergent
define { i64, i64 } @_RINvXs4_NtCsiWbdGYq5HZh_11cuda_device6threadNtB6_7Index1DNtB6_12IndexFormula10from_scopeNtNtB6_10___internal13UnknownDomainNtB1q_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0) #2 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i64 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v1) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v3 = icmp eq i64 %v2, -1
  br i1 %v3, label %bb4, label %bb3

bb2:                                              ; preds = %bb4, %bb3
  %v4 = phi i64 [ %v10, %bb3 ], [ %v13, %bb4 ]
  %v5 = phi i64 [ %v11, %bb3 ], [ %v14, %bb4 ]
  %v6 = insertvalue { i64, i64 } undef, i64 %v4, 0
  %v7 = insertvalue { i64, i64 } %v6, i64 %v5, 1
  ret { i64, i64 } %v7

bb3:                                              ; preds = %bb1
  %v8 = insertvalue { i64, i64 } undef, i64 1, 0
  %v9 = insertvalue { i64, i64 } %v8, i64 %v2, 1
  %v10 = extractvalue { i64, i64 } %v9, 0
  %v11 = extractvalue { i64, i64 } %v9, 1
  br label %bb2

bb4:                                              ; preds = %bb1
  %v12 = insertvalue { i64, i64 } undef, i64 0, 0
  %v13 = extractvalue { i64, i64 } %v12, 0
  %v14 = extractvalue { i64, i64 } %v12, 1
  br label %bb2
}

; Function Attrs: alwaysinline convergent
define i64 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0) #2 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2

bb2:                                              ; preds = %bb1
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3

bb3:                                              ; preds = %bb2
  %v5 = zext i32 %v2 to i64
  %v6 = zext i32 %v3 to i64
  %v7 = zext i32 %v4 to i64
  %v8 = icmp eq i64 %v6, 0
  br i1 %v8, label %bb10, label %bb8

bb4:                                              ; preds = %bb12
  %v9 = xor i1 %v20, true
  br i1 %v9, label %bb6, label %bb5

bb5:                                              ; preds = %bb4
  %v10 = icmp ne i64 %v19, -1
  br label %bb7

bb6:                                              ; preds = %bb4
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  %v11 = phi i1 [ %v10, %bb5 ], [ false, %bb6 ]
  %v12 = xor i1 %v11, true
  br i1 %v12, label %bb14, label %bb13

bb8:                                              ; preds = %bb3
  %v13 = sub i64 -1, %v7
  %v14 = udiv i64 %v13, %v6
  %v15 = icmp ugt i64 %v5, %v14
  %v16 = xor i1 %v15, true
  br i1 %v16, label %bb11, label %bb9

bb9:                                              ; preds = %bb8
  br label %bb10

bb10:                                             ; preds = %bb9, %bb3
  br label %bb12

bb11:                                             ; preds = %bb8
  %v17 = mul i64 %v5, %v6
  %v18 = add i64 %v17, %v7
  br label %bb12

bb12:                                             ; preds = %bb11, %bb10
  %v19 = phi i64 [ -1, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v1) #0
  br label %bb4

bb13:                                             ; preds = %bb7
  %v21 = icmp eq i64 %v19, -1
  br i1 %v21, label %bb14, label %bb15

bb14:                                             ; preds = %bb13, %bb7
  br label %bb15

bb15:                                             ; preds = %bb14, %bb13
  %v22 = phi i64 [ %v19, %bb13 ], [ -1, %bb14 ]
  ret i64 %v22
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: alwaysinline convergent
define i1 @_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsg09vZ5ZKBJe_16nearest_neighbor(ptr %v0) #2 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1
  %v3 = xor i1 %v2, true
  br i1 %v3, label %bb2, label %bb1

bb1:                                              ; preds = %bb0
  br label %bb8

bb2:                                              ; preds = %bb0
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb3

bb3:                                              ; preds = %bb2
  %v5 = icmp eq i32 %v4, 1
  br i1 %v5, label %bb4, label %bb5

bb4:                                              ; preds = %bb3
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  br label %bb6

bb5:                                              ; preds = %bb3
  br label %bb7

bb6:                                              ; preds = %bb4
  %v7 = icmp eq i32 %v6, 1
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  %v8 = phi i1 [ false, %bb5 ], [ %v7, %bb6 ]
  br label %bb8

bb8:                                              ; preds = %bb7, %bb1
  %v9 = phi i1 [ true, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, true
  br i1 %v10, label %bb9, label %bb10

bb9:                                              ; preds = %bb8
  %v11 = icmp eq i8 0, 2
  %v12 = xor i1 %v11, true
  br i1 %v12, label %bb11, label %bb10

bb10:                                             ; preds = %bb9, %bb8
  br label %bb17

bb11:                                             ; preds = %bb9
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb12

bb12:                                             ; preds = %bb11
  %v14 = icmp eq i32 %v13, 1
  br i1 %v14, label %bb13, label %bb14

bb13:                                             ; preds = %bb12
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb15

bb14:                                             ; preds = %bb12
  br label %bb16

bb15:                                             ; preds = %bb13
  %v16 = icmp eq i32 %v15, 1
  br label %bb16

bb16:                                             ; preds = %bb15, %bb14
  %v17 = phi i1 [ false, %bb14 ], [ %v16, %bb15 ]
  br label %bb17

bb17:                                             ; preds = %bb16, %bb10
  %v18 = phi i1 [ true, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, true
  br i1 %v19, label %bb19, label %bb18

bb18:                                             ; preds = %bb17
  br label %bb20

bb19:                                             ; preds = %bb17
  br label %bb20

bb20:                                             ; preds = %bb19, %bb18
  %v20 = phi i1 [ %v18, %bb18 ], [ false, %bb19 ]
  ret i1 %v20
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #3

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
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { alwaysinline convergent }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { alwaysinline nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #7 = { nounwind }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
