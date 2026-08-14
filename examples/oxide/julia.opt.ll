; ModuleID = '/home/kranex/dev/rust-metaprogramming/examples/oxide/julia.ll'
source_filename = "julia"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @julia], section "llvm.metadata"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0

; Function Attrs: convergent nofree norecurse nosync nounwind memory(write, inaccessiblemem: none)
define ptx_kernel void @julia(ptr writeonly captures(address) %v0, i64 %v1, i32 %v2) #1 {
entry:
  %v6.i = zext i32 %v2 to i64
  %v7.i = icmp eq i32 %v2, 0
  br i1 %v7.i, label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCsfJW2dUZXFMg_5julia4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, label %bb24.i

bb3.i:                                            ; preds = %bb24.i
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2
  %v17.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2
  %v18.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2
  %narrow.i = mul nuw nsw i32 %v13.i, %v17.i
  %narrow3.i = add nuw nsw i32 %narrow.i, %v18.i
  %v68.i = zext nneg i32 %narrow3.i to i64
  %v70.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
  %v23.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2
  %v24.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
  %v25.i = zext nneg i32 %v70.i to i64
  %v26.i = zext nneg i32 %v23.i to i64
  %v27.i = zext nneg i32 %v24.i to i64
  %v75.i = mul nuw nsw i64 %v25.i, %v26.i
  %v76.i = add nuw nsw i64 %v75.i, %v27.i
  %v30.not.i.not = icmp samesign ult i64 %v76.i, %v6.i
  %v42.i = shl nuw nsw i64 %v68.i, 32
  %v43.i = or i64 %v42.i, %v76.i
  %spec.select12 = select i1 %v30.not.i.not, i64 %v43.i, i64 undef
  br label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCsfJW2dUZXFMg_5julia4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit

bb24.i:                                           ; preds = %entry
  %v57.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #2
  %v58.i = icmp ne i32 %v57.i, 1
  %v59.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #2
  %v60.i = icmp ne i32 %v59.i, 1
  %v61.i = or i1 %v58.i, %v60.i
  br i1 %v61.i, label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCsfJW2dUZXFMg_5julia4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, label %bb3.i

_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCsfJW2dUZXFMg_5julia4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit: ; preds = %bb3.i, %entry, %bb24.i
  %v51.i = phi i1 [ false, %entry ], [ false, %bb24.i ], [ %v30.not.i.not, %bb3.i ]
  %v52.i = phi i64 [ undef, %entry ], [ undef, %bb24.i ], [ %spec.select12, %bb3.i ]
  br i1 %v51.i, label %bb3, label %bb21

bb3:                                              ; preds = %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCsfJW2dUZXFMg_5julia4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit
  %v20 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
  %v69 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2
  %v71 = mul i32 %v20, %v69
  %v72 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
  %v76 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2
  %v78 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2
  %narrow = mul nuw nsw i32 %v76, %v78
  %v81 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2
  %0 = add i32 %v71, %v72
  %v85 = sub i32 %v2, %0
  %v86 = uitofp i32 %v85 to float
  %v87 = fmul contract float %v86, 0x3FB99999A0000000
  %v88 = uitofp i32 %v2 to float
  %v89 = fdiv contract float %v87, %v88
  %1 = add nuw nsw i32 %narrow, %v81
  %v90 = sub i32 %v2, %1
  %v91 = uitofp i32 %v90 to float
  %v92 = fmul contract float %v91, 0x3FB99999A0000000
  %v93 = fdiv contract float %v92, %v88
  br label %bb31

bb7:                                              ; preds = %bb14
  %v10.i = and i64 %v52.i, 4294967295
  %v14.not.i = icmp samesign ult i64 %v10.i, %v6.i
  br i1 %v14.not.i, label %bb5.i, label %_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCsfJW2dUZXFMg_5julia4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_.exit

bb5.i:                                            ; preds = %bb7
  %v9.i = lshr i64 %v52.i, 32
  %v22.i = mul nuw nsw i64 %v9.i, %v6.i
  %v23.i9 = add nuw nsw i64 %v22.i, %v10.i
  %v26.not.i = icmp ult i64 %v23.i9, %v1
  %v30.i = getelementptr inbounds nuw { [4 x float] }, ptr %v0, i64 %v23.i9
  %v36.i = select i1 %v26.not.i, ptr %v30.i, ptr null
  br label %_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCsfJW2dUZXFMg_5julia4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_.exit

_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCsfJW2dUZXFMg_5julia4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_.exit: ; preds = %bb7, %bb5.i
  %v39.i = phi ptr [ %v36.i, %bb5.i ], [ null, %bb7 ]
  %v55.not = icmp eq ptr %v39.i, null
  br i1 %v55.not, label %bb21, label %bb17

bb8:                                              ; preds = %bb31
  %v30 = fmul contract float %v2113, %v2113
  %v31 = fmul contract float %v2214, %v2214
  %v32 = fsub contract float %v30, %v31
  %v33 = fadd contract float %v32, 0xBFE99999A0000000
  %v34 = fmul contract float %v2113, %v2214
  %v36 = fadd contract float %v34, %v34
  %v37 = fadd contract float %v36, 0x3FC3F7CEE0000000
  %v38 = fmul contract float %v33, %v33
  %v39 = fmul contract float %v37, %v37
  %v40 = fadd contract float %v38, %v39
  %v41 = fcmp ogt float %v40, 1.000000e+03
  %v23. = select i1 %v41, float 0.000000e+00, float %v2315
  %v46 = select i1 %v41, float %v2113, float %v33
  %v47 = select i1 %v41, float %v2214, float %v37
  br label %bb14

bb14:                                             ; preds = %bb8, %bb31
  %v48 = phi float [ %v2113, %bb31 ], [ %v46, %bb8 ]
  %v49 = phi float [ %v2214, %bb31 ], [ %v47, %bb8 ]
  %v50 = phi float [ %v2315, %bb31 ], [ %v23., %bb8 ]
  %v51 = phi i1 [ true, %bb31 ], [ %v41, %bb8 ]
  %v26 = icmp samesign ult i32 %v10217, 200
  %v94 = zext i1 %v26 to i32
  %v102 = add nuw nsw i32 %v10217, %v94
  br i1 %v26, label %bb31, label %bb7

bb17:                                             ; preds = %_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCsfJW2dUZXFMg_5julia4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_.exit
  %v52.repack6 = getelementptr inbounds nuw i8, ptr %v39.i, i64 12
  %v52.repack4 = getelementptr inbounds nuw i8, ptr %v39.i, i64 8
  %v52.repack2 = getelementptr inbounds nuw i8, ptr %v39.i, i64 4
  %v61 = fmul contract float %v50, 2.550000e+02
  store float %v61, ptr %v39.i, align 16
  store float 0.000000e+00, ptr %v52.repack2, align 4
  store float 0.000000e+00, ptr %v52.repack4, align 8
  store float 2.550000e+02, ptr %v52.repack6, align 4
  br label %bb21

bb21:                                             ; preds = %_RNvMst_NtCsiWbdGYq5HZh_11cuda_device8disjointINtB5_13DisjointSliceNtCsfJW2dUZXFMg_5julia4RgbaNtNtB7_6thread14Runtime2DIndexE7get_mutB14_.exit, %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCsfJW2dUZXFMg_5julia4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, %bb17
  ret void

bb31:                                             ; preds = %bb3, %bb14
  %v10217 = phi i32 [ 1, %bb3 ], [ %v102, %bb14 ]
  %v2416 = phi i1 [ false, %bb3 ], [ %v51, %bb14 ]
  %v2315 = phi float [ 1.000000e+00, %bb3 ], [ %v50, %bb14 ]
  %v2214 = phi float [ %v93, %bb3 ], [ %v49, %bb14 ]
  %v2113 = phi float [ %v89, %bb3 ], [ %v48, %bb14 ]
  br i1 %v2416, label %bb14, label %bb8
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nofree norecurse nosync nounwind memory(write, inaccessiblemem: none) }
attributes #2 = { convergent }
