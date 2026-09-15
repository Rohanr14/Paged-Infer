// Candidate accumulator setup and main loop: target/release/deps/paged_infer-64c151c92726cdc3.s, lines 116568-116622
	b.ne	LBB296_150
	b	LBB296_70
LBB296_81:
	mov	x20, #0
	movi.2d	v0, #0000000000000000
	madd	x0, x14, x28, x16
	movi.2d	v1, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	movi.2d	v5, #0000000000000000
	movi.2d	v6, #0000000000000000
	madd	x2, x14, x26, x16
	movi.2d	v7, #0000000000000000
	movi.2d	v16, #0000000000000000
	movi.2d	v17, #0000000000000000
	movi.2d	v18, #0000000000000000
	movi.2d	v19, #0000000000000000
	movi.2d	v20, #0000000000000000
	movi.2d	v4, #0000000000000000
	movi.2d	v21, #0000000000000000
	movi.2d	v22, #0000000000000000
	movi.2d	v23, #0000000000000000
LBB296_82:
	lsl	x19, x20, #2
	add	x21, x8, x19
	add	x19, x3, x19
	ldp	q24, q25, [x19]
	ldp	q26, q27, [x0, #-32]
	fmla.4s	v23, v26, v24
	ldp	q28, q29, [x2, #-32]
	fmla.4s	v20, v28, v24
	ldp	q24, q30, [x21]
	fmla.4s	v16, v26, v24
	fmla.4s	v3, v28, v24
	fmla.4s	v22, v27, v25
	fmla.4s	v19, v29, v25
	fmla.4s	v7, v27, v30
	fmla.4s	v2, v29, v30
	ldp	q24, q25, [x19, #32]
	ldp	q26, q27, [x0], #64
	fmla.4s	v21, v26, v24
	ldp	q28, q29, [x2], #64
	fmla.4s	v18, v28, v24
	ldp	q24, q30, [x21, #32]
	fmla.4s	v6, v26, v24
	fmla.4s	v1, v28, v24
	fmla.4s	v4, v27, v25
	fmla.4s	v17, v29, v25
	fmla.4s	v5, v27, v30
	add	x19, x20, #16
	add	x21, x20, #32
	fmla.4s	v0, v29, v30
	mov	x20, x19
	cmp	x21, x14
	b.ls	LBB296_82

// Candidate vector remainder and no-scalar-tail reduction: target/release/deps/paged_infer-64c151c92726cdc3.s, lines 116630-116695
	lsl	x0, x19, #2
	madd	x2, x14, x26, x0
	ldr	x20, [sp, #464]
	add	x2, x20, x2
	madd	x0, x14, x28, x0
	add	x20, x20, x0
LBB296_86:
	ldr	q24, [x20], #16
	ldr	q25, [x2], #16
	lsl	x0, x19, #2
	ldr	q26, [x3, x0]
	ldr	q27, [x8, x0]
	fmla.4s	v23, v24, v26
	fmla.4s	v20, v25, v26
	fmla.4s	v16, v24, v27
	fmla.4s	v3, v25, v27
	add	x0, x19, #4
	add	x21, x19, #8
	mov	x19, x0
	cmp	x21, x14
	b.ls	LBB296_86
LBB296_87:
	fadd.4s	v22, v22, v23
	fadd.4s	v4, v4, v21
	fadd.4s	v4, v4, v22
	faddp.4s	v4, v4, v4
	faddp.2s	s4, v4
	subs	x2, x14, x0
	b.ls	LBB296_90
	cmp	x2, #3
	b.hi	LBB296_91
	mov	x20, x0
	b	LBB296_102
LBB296_90:
	fadd.4s	v19, v19, v20
	fadd.4s	v17, v17, v18
	fadd.4s	v17, v17, v19
	faddp.4s	v17, v17, v17
	faddp.2s	s17, v17
	fmov	w8, s4
	fmov	w14, s17
	orr	x8, x8, x14, lsl #32
	fadd.4s	v4, v7, v16
	fadd.4s	v5, v5, v6
	fadd.4s	v4, v5, v4
	faddp.4s	v4, v4, v4
	faddp.2s	s5, v4
	fadd.4s	v2, v2, v3
	fadd.4s	v0, v0, v1
	fadd.4s	v0, v0, v2
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	ldp	x21, x20, [sp, #424]
	fmov	s2, w8
	lsr	x8, x8, #32
	fmov	s1, w8
	subs	x14, x20, x30
	b.eq	LBB296_70
	b	LBB296_150
LBB296_91:
	cmp	x2, #16
	b.hs	LBB296_93
	mov	x19, #0
	b	LBB296_98
LBB296_93:
	mov	x20, #0

// Existing four-query single-key main loop for comparison: target/release/deps/paged_infer-64c151c92726cdc3.s, lines 117476-117511
LBB296_193:
	add	x20, x3, x20, lsl #2
	ldp	q24, q25, [x20]
	ldp	q26, q27, [x2, #-32]
	fmla.4s	v23, v26, v24
	ldp	q26, q28, [x0, #-32]
	fmla.4s	v20, v26, v24
	ldp	q26, q29, [x15, #-32]
	fmla.4s	v17, v26, v24
	ldp	q26, q30, [x14, #-32]
	fmla.4s	v4, v26, v24
	fmla.4s	v22, v27, v25
	fmla.4s	v19, v28, v25
	fmla.4s	v16, v29, v25
	fmla.4s	v3, v30, v25
	ldp	q24, q25, [x20, #32]
	ldp	q26, q27, [x2], #64
	fmla.4s	v21, v26, v24
	ldp	q26, q28, [x0], #64
	fmla.4s	v18, v26, v24
	ldp	q26, q29, [x15], #64
	fmla.4s	v7, v26, v24
	ldp	q26, q30, [x14], #64
	fmla.4s	v2, v26, v24
	fmla.4s	v0, v27, v25
	fmla.4s	v5, v28, v25
	fmla.4s	v6, v29, v25
	mov	x20, x19
	add	x19, x19, #16
	fmla.4s	v1, v30, v25
	cmp	x19, x9
	b.ls	LBB296_193
	and	x15, x9, #0x1ffffffffffffff0
	orr	x14, x15, #0x4
	cmp	x14, x9
	b.ls	LBB296_190
