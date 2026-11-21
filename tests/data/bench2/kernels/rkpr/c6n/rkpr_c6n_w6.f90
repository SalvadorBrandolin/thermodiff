subroutine rkpr_c6n_w6(nc, V, T, Tc, ac, b, delta1, k, kij, lij, n, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, &
     dfdndT, dfdndV)
   use yaeos__constants, only: R
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), intent(in) :: Tc(:)
   real(8), intent(in) :: ac(:)
   real(8), intent(in) :: b(:)
   real(8), intent(in) :: delta1(:)
   real(8), intent(in) :: k(:)
   real(8), intent(in) :: kij(:, :)
   real(8), intent(in) :: lij(:, :)
   real(8), intent(in) :: n(:)
   real(8), intent(out) :: f
   real(8), intent(out) :: dfdT
   real(8), intent(out) :: dfdV
   real(8), intent(out) :: dfdn(:)
   real(8), intent(out) :: dfdT2
   real(8), intent(out) :: dfdV2
   real(8), intent(out) :: dfdndn(:, :)
   real(8), intent(out) :: dfdTdV
   real(8), intent(out) :: dfdndT(:)
   real(8), intent(out) :: dfdndV(:)
   real(8) :: res, t10, t100, t101, t102, t103, t104, t105, t106, t109, t110, t111, t112, t113, t114, t115, t12, t13, t14
   real(8) :: t15, t16, t17, t18, t19, t2, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t3, t30, t31, t32, t33, t34
   real(8) :: t35, t36, t37, t38, t39, t4, t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t5, t50, t51, t52, t53, t54
   real(8) :: t55, t56, t57, t58, t59, t6, t60, t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75
   real(8) :: t76, t77, t78, t79, t80, t81, t82, t83, t84, t85, t86, t87, t88, t89, t9, t90, t91, t92, t93, t94, t95, t96
   real(8) :: t97, t98, t99
   real(8) :: cmp0(nc), cmp0_T(nc), cmp0_TT(nc), mat1_g(nc), mat1_gT(nc), t1(nc), t107(nc), t108(nc), t11(nc), t7(nc)
   real(8) :: t8(nc)
   real(8) :: mat1(nc, nc), mat2(nc, nc)
   integer :: i, ia, ib, ic, j, p, q, s, u

   do ic = 1, nc
      cmp0(ic) = (3.0d0/(T/Tc(ic) + 2.0d0))**k(ic)*ac(ic)
   end do

   do ic = 1, nc
      cmp0_T(ic) = -cmp0(ic)*k(ic)/((T/Tc(ic) + 2.0d0)*Tc(ic))
   end do

   do ic = 1, nc
      cmp0_TT(ic) = -cmp0_T(ic)*k(ic)/((T/Tc(ic) + 2.0d0)*Tc(ic)) + cmp0(ic)*k(ic)/((T/Tc(ic &
      ) + 2.0d0)**2*Tc(ic)**2)
   end do

   do ic = 1, nc
      mat1_g(ic) = (1.0d0/2.0d0)*cmp0_T(ic)/cmp0(ic)
   end do

   do ic = 1, nc
      mat1_gT(ic) = (1.0d0/2.0d0)*cmp0_TT(ic)/cmp0(ic) - 1.0d0/2.0d0*cmp0_T(ic)**2/cmp0(ic) &
      **2
   end do

   do ia = 1, nc
      do ib = ia, nc
         mat1(ia, ib) = sqrt(cmp0(ia)*cmp0(ib))*(1.0d0 - kij(ia, ib))
         mat1(ib, ia) = mat1(ia, ib)
      end do
   end do

   do ia = 1, nc
      do ib = ia, nc
         mat2(ia, ib) = (1.0d0 - lij(ia, ib))*(0.5d0*b(ia) + 0.5d0*b(ib))
         mat2(ib, ia) = mat2(ia, ib)
      end do
   end do

   do p = 1, nc
      t1(p) = delta1(p)*n(p)
   end do

   do s = 1, nc
      t7(s) = 0.0d0
      do q = 1, nc
         t7(s) = t7(s) + mat1(q, s)*n(q)
      end do
   end do

   do s = 1, nc
      t8(s) = 0.0d0
      do q = 1, nc
         t8(s) = t8(s) + mat2(q, s)*n(q)
      end do
   end do

   t9 = 0.0d0
   do q = 1, nc
      t9 = t9 + n(q)
   end do

   t10 = 0.0d0
   do q = 1, nc
      t10 = t10 + t1(q)
   end do

   do s = 1, nc
      t11(s) = 0.0d0
      do q = 1, nc
         t11(s) = t11(s) + (mat1_g(q) + mat1_g(s))*mat1(q, s)*n(q)
      end do
   end do

   t12 = 0.0d0
   do q = 1, nc
      t12 = t12 + (2*mat1(q, q)*mat1_gT(q) + 4*mat1(q, q)*mat1_g(q)**2)*n(q)**2
   end do

   t13 = 0.0d0
   do q = 1, nc
      do s = q + 1, nc
         t13 = t13 + ((mat1_gT(q) + mat1_gT(s))*mat1(q, s) + (mat1_g(q) + mat1_g(s))**2*mat1( &
      q, s))*n(q)*n(s)
      end do
   end do

   t14 = 0.0d0
   do s = 1, nc
      t14 = t14 + n(s)*t7(s)
   end do

   t15 = 0.0d0
   do s = 1, nc
      t15 = t15 + n(s)*t8(s)
   end do

   t16 = 0.0d0
   do s = 1, nc
      t16 = t16 + n(s)*t11(s)
   end do

   t17 = 1d0/t9

   t18 = t10*t17

   t19 = t18 + 1.0d0

   t20 = -t10*t17 + 1.0d0

   t21 = 1d0/t19

   t22 = t9**(-2)

   t23 = V + t15*t17*t20*t21

   t24 = V + t10*t15*t22

   t25 = t19**(-2)

   t26 = t18 - t20*t21

   t27 = t9**(-3)

   t28 = 1d0/t23

   t29 = t23**(-2)

   t30 = 1d0/t15

   t31 = 1d0/t26

   t32 = 1d0/t24

   t33 = log(t24*t28)

   t34 = t26**(-2)

   t35 = t15*t20*t22*t25 + t15*t21*t22

   t36 = -t10*t15*t20*t25*t27 - t10*t15*t21*t27 + t15*t20*t21*t22

   t37 = 1d0/V

   t38 = t10*t22*t28 - t17*t20*t21*t24*t29

   t39 = -t15*t17*t37 + 1.0d0

   t40 = t15*t22*t28 + t24*t29*t35

   t41 = -2*t10*t15*t27*t28 + t24*t29*t36

   t42 = t15**(-2)

   t43 = -t17*t20*t25 - t17*t21 - t17

   t44 = t10*t20*t22*t25 + t10*t21*t22 + t10*t22

   t45 = t24**(-2)

   t46 = t23**(-3)

   t47 = t9**(-4)

   t48 = 1d0/t39

   t49 = t19**(-3)

   t50 = -t24*t29 + t28

   t51 = t39**(-2)

   t52 = V**(-2)

   t53 = t18 - 1.0d0

   t54 = t26**(-3)

   t55 = log(t39)

   t56 = t10**2

   t57 = -2*t10*t15*t20*t25*t27 - 2*t10*t15*t21*t27 + 2*t15*t20*t21*t22

   t58 = -t23*t30*t31*t32*t41*t9 - t30*t31*t33 - t30*t33*t34*t44*t9

   t59 = t10*t15*t20*t25*t27 + t10*t15*t21*t27 - t15*t20*t21*t22

   t60 = -t23*t30*t31*t32*t40*t9 - t30*t33*t34*t43*t9

   t61 = -t23*t30*t31*t32*t38*t9 + t31*t33*t42*t9

   t62 = 2*t15*t20*t22*t25 + 2*t15*t21*t22

   t63 = -t15*t20*t22*t25 - t15*t21*t22

   t64 = 1d0/(V - t15*t17*t21*t53)

   t65 = t20*t22*t25

   t66 = V**(-3)

   t67 = t21*t22

   t68 = t15**2

   t69 = t24*t29*(2*t10*t15*t20*t47*t49 + 2*t10*t15*t25*t47 - 2*t15*t20*t25*t27 - &
      2*t15*t21*t27)

   t70 = -t14*t30*t33*t34*t9*(-2*t10*t20*t27*t49 - 2*t10*t25*t27 + t22 + t65 + &
      t67)

   t71 = R*T*t37*t48 - t14*t23*t30*t31*t32*t38*t9 + t14*t31*t33*t42*t9

   t72 = 2*t10*t20*t22*t25 + 2*t10*t21*t22 + 2*t10*t22

   t73 = -2*t17*t20*t25 - 2*t17*t21 - 2*t17

   t74 = -t14*t23*t30*t32*t34*t41*t43*t9

   t75 = -t14*t23*t30*t32*t34*t40*t44*t9

   t76 = -t14*t23*t30*t32*t34*t38*t44*t9

   t77 = -t14*t23*t30*t32*t34*t38*t43*t9

   t78 = t14*t23*t31*t32*t41*t42*t9

   t79 = t14*t23*t31*t32*t40*t42*t9

   t80 = -t14*t23*t30*t31*t32*t40

   t81 = -t14*t23*t30*t31*t32*t38

   t82 = 2*t10*t15*t20*t21*t29*t47

   t83 = -R*T*t15*t22*t51*t52

   t84 = -2*t10*t15*t27*t29*t35

   t85 = t14*t33*t34*t42*t44*t9

   t86 = t14*t33*t34*t42*t43*t9

   t87 = -t15*t20*t21*t27*t29

   t88 = -t14*t30*t33*t34*t43

   t89 = -2*t15*t27*t28

   t90 = -2*t10*t27*t28

   t91 = 1d0/(t18 + t21*t53)

   t92 = t15*t22*t29*t36

   t93 = t14*t31*t33*t42

   t94 = t10*t22*t29*t36

   t95 = t10*t22*t29*t35

   t96 = -t24*t64 + 1

   t97 = t9**(-5)

   t98 = t22*t28

   t99 = -t14*t23*t30*t31*t32*t40*t9 - t14*t30*t33*t34*t43*t9

   t100 = t30*t31*t33*t9

   t101 = -R*T*t15*t17*t37*t48 - R*T*t55 - t14*t23*t30*t31*t32*t41*t9 - t14*t30* &
      t31*t33 - t14*t30*t33*t34*t44*t9

   t102 = 2*t14*t17*t23*t31*t38*t45 - 2*t14*t23*t30*t31*t32*t9*(-t17*t20*t21*t24* &
      t46*t62 + t20*t22*t24*t25*t29 + t21*t22*t24*t29 + t87 + t95 + t98 &
      ) - 2*t14*t30*t31*t32*t38*t63*t9 + 2*t77 + 2*t79 + 2*t86

   t103 = 2*t10*t14*t17*t23*t30*t31*t40*t45 - 2*t14*t20*t21*t30*t31*t32*t40 - 2* &
      t14*t23*t30*t31*t32*t9*(-2*t17*t20*t21*t24*t35*t46 + t24*t29*(t65 &
      + t67) + t87 + t95 + t98) + 2*t77 + 2*t79 + 2*t86

   t104 = 4*R*T*t17*t51*t52 + 4*t10*t14*t17*t23*t30*t31*t38*t45 - 4*t14*t20*t21* &
      t30*t31*t32*t38 - 4*t14*t23*t30*t31*t32*t9*(-2*t10*t20*t21*t27* &
      t29 + 2*t20**2*t22*t24*t25*t46) + 8*t14*t23*t31*t32*t38*t42*t9 - &
      8*t14*t31*t33*t9/t15**3

   t105 = t14*t17*t23*t31*t40*t45 - t14*t23*t30*t31*t32*t9*(2*t15*t22*t29*t35 + &
      t24*t29*(-2*t15*t20*t27*t49 - 2*t15*t25*t27) + t24*t35*t46*t62) - &
      2*t14*t23*t30*t32*t34*t40*t43*t9 - t14*t30*t31*t32*t40*t63*t9 - &
      t14*t30*t33*t34*t9*(2*t20*t22*t49 + 2*t22*t25) - t14*t30*t33*t43* &
      t54*t73*t9

   t106 = R*T*t27*t51*t52*t68 - 2*t10*t14*t22*t23*t31*t41*t45 - 2*t14*t23*t30*t31* &
      t32*t41 - t14*t23*t30*t31*t32*t9*(-4*t10*t15*t27*t29*t36 + 6*t10* &
      t15*t28*t47 + t24*t29*(4*t10*t15*t20*t25*t47 + 4*t10*t15*t21*t47 &
      - 2*t15*t20*t21*t27 - 2*t15*t20*t49*t56*t97 - 2*t15*t25*t56*t97) &
      + t24*t36*t46*t57) - 2*t14*t23*t30*t32*t34*t41*t44*t9 - t14*t30* &
      t31*t32*t41*t59*t9 - 2*t14*t30*t33*t34*t44 - t14*t30*t33*t34*t9*( &
      -2*t10*t20*t25*t27 - 2*t10*t21*t27 - 2*t10*t27 + 2*t20*t47*t49* &
      t56 + 2*t25*t47*t56) - t14*t30*t33*t44*t54*t72*t9

   do u = 1, nc
      t107(u) = 2*t58*t7(u) + (-4*t10*t14*t22*t23*t31*t38*t45 - 2*t14*t23*t30*t31*t32*t9 &
      *(-t10*t20*t24*t25*t27*t29 - t10*t21*t24*t27*t29 - t17*t20*t21* &
      t24*t46*t57 + t20*t21*t22*t24*t29 + t82 + t90 + t94) - 2*t14*t30* &
      t31*t32*t38*t59*t9 + 2*t76 + 2*t78 + 2*t81 + 2*t83 + 2*t85 + 2* &
      t93)*t8(u) + (-2*t10*t14*t22*t23*t31*t40*t45 - t14*t23*t30*t31* &
      t32*t9*(t24*t35*t46*t57 + t69 + t84 + t89 + t92) - t14*t30*t31* &
      t32*t40*t59*t9 - t14*t30*t33*t43*t54*t72*t9 + t70 + t74 + t75 + &
      t80 + t88)*delta1(u)
   end do

   do u = 1, nc
      t108(u) = 2*t58*t7(u) + (t14*t17*t23*t31*t41*t45 - t14*t23*t30*t31*t32*t9*(t24*t36 &
      *t46*t62 + t69 + t84 + t89 + t92) - t14*t30*t31*t32*t41*t63*t9 - &
      t14*t30*t33*t44*t54*t73*t9 + t70 + t74 + t75 + t80 + t88)*delta1( &
      u) + (2*t10*t14*t17*t23*t30*t31*t41*t45 - 2*t14*t20*t21*t30*t31* &
      t32*t41 - 2*t14*t23*t30*t31*t32*t9*(-2*t17*t20*t21*t24*t36*t46 + &
      t24*t29*(-t10*t20*t25*t27 - t10*t21*t27 + t20*t21*t22) + t82 + &
      t90 + t94) + 2*t76 + 2*t78 + 2*t81 + 2*t83 + 2*t85 + 2*t93)*t8(u)
   end do

   t109 = 2*R*t37*t48 + 2*t16*t61

   t110 = t16*t60

   t111 = -R*t15*t17*t37*t48 - R*t55 + t16*t58

   t112 = -t14*t23*t30*t31*t32*t9*(-t15*t22*t29 - 2*t24*t35*t46 + t29*t35) + t14* &
      t23*t30*t31*t40*t45*t9 - t14*t23*t30*t32*t34*t43*t50*t9 - t14*t30 &
      *t31*t32*t40*t9

   t113 = -2*R*T*t15*t17*t51*t66 - 2*R*T*t48*t52 - 2*t14*t23*t30*t31*t32*t9*(-t10* &
      t22*t29 + 2*t17*t20*t21*t24*t46 - t17*t20*t21*t29) + 2*t14*t23* &
      t30*t31*t38*t45*t9 + 2*t14*t23*t31*t32*t42*t50*t9 - 2*t14*t30*t31 &
      *t32*t38*t9

   t114 = t23*t30*t31*t32*t50*t9

   t115 = R*T*t22*t51*t66*t68 - t14*t23*t30*t31*t32*t50 - t14*t23*t30*t31*t32*t9*( &
      2*t10*t15*t27*t29 - 2*t24*t36*t46 + t29*t36) + t14*t23*t30*t31* &
      t41*t45*t9 - t14*t23*t30*t32*t34*t44*t50*t9 - t14*t30*t31*t32*t41 &
      *t9

   res = -R*T*t55*t9 - t14*t30*t31*t33*t9
   f = res

   res = -R*t55*t9 - t16*t30*t31*t33*t9
   dfdT = res

   res = -R*T*t15*t48*t52 - t14*t23*t30*t31*t32*t50*t9
   dfdV = res

   do i = 1, nc
      res = -2*t100*t7(i) + t101 + 2*t71*t8(i) + t99*delta1(i)
      dfdn(i) = res
   end do

   res = -t30*t31*t33*t9*(t12 + 2*t13)
   dfdT2 = res

   res = 2*R*T*t15*t48*t66 + R*T*t17*t51*t68/V**4 + t14*t30*t32*t64*t9*t91* &
         t96 + t14*t30*t45*t9*t91*t96
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = -2*t100*mat1(i, j) + t102*delta1(j)*t8(i) + t103*delta1(i)*t8(j) + &
               t104*t8(i)*t8(j) + t105*delta1(i)*delta1(j) + t106 + 2*t60*delta1 &
               (i)*t7(j) + 2*t60*delta1(j)*t7(i) + 4*t61*t7(i)*t8(j) + 4*t61*t7( &
               j)*t8(i) + 2*t71*mat2(i, j) + t107(i) + t108(j)
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t15*t48*t52 - t16*t23*t30*t31*t32*t50*t9
   dfdTdV = res

   do i = 1, nc
      res = -2*t100*t11(i) + t109*t8(i) + t110*delta1(i) + t111
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = t112*delta1(i) + t113*t8(i) - 2*t114*t7(i) + t115
      dfdndV(i) = res
   end do

end subroutine rkpr_c6n_w6
