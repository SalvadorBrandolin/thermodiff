subroutine rkpr_c6m_w6(nc, V, T, Tc, ac, b, delta1, k, kij, lij, n, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, &
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
   real(8) :: res, t10, t100, t101, t102, t103, t104, t105, t106, t107, t11, t110, t111, t112, t113, t114, t115, t116, t117
   real(8) :: t118, t119, t13, t14, t15, t16, t17, t18, t19, t2, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t3, t30
   real(8) :: t31, t32, t33, t34, t35, t36, t37, t38, t39, t4, t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t5, t50
   real(8) :: t51, t52, t53, t54, t55, t56, t57, t58, t59, t6, t60, t61, t62, t63, t64, t65, t66, t67, t68, t69, t7, t70
   real(8) :: t71, t72, t73, t74, t75, t76, t77, t78, t79, t80, t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92
   real(8) :: t93, t94, t95, t96, t97, t98, t99
   real(8) :: cmp0(nc), cmp0_T(nc), cmp0_TT(nc), cmp1(nc), cmp1_T(nc), cmp1_TT(nc), t1(nc), t108(nc), t109(nc), t12(nc)
   real(8) :: t8(nc), t9(nc)
   real(8) :: mat2(nc, nc)
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
      cmp1(ic) = sqrt(cmp0(ic))
   end do

   do ic = 1, nc
      cmp1_T(ic) = (1.0d0/2.0d0)*cmp0_T(ic)*cmp1(ic)/cmp0(ic)
   end do

   do ic = 1, nc
      cmp1_TT(ic) = ((1.0d0/2.0d0)*cmp0_TT(ic)/cmp0(ic) - 1.0d0/2.0d0*cmp0_T(ic)**2/cmp0(ic) &
      **2)*cmp1(ic) + (1.0d0/2.0d0)*cmp0_T(ic)*cmp1_T(ic)/cmp0(ic)
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
      t8(s) = 0.0d0
   end do
   do s = 1, nc
      t117 = 0.0d0
      do q = 1, s - 1
         t8(q) = t8(q) + mat2(q, s)*n(s)
         t117 = t117 + mat2(q, s)*n(q)
      end do
      t8(s) = t8(s) + t117 + mat2(s, s)*n(s)
   end do

   do s = 1, nc
      t9(s) = 0.0d0
   end do
   do s = 1, nc
      t118 = 0.0d0
      do q = 1, s - 1
         t9(q) = t9(q) + (1.0d0 - kij(q, s))*cmp1(s)*n(s)
         t118 = t118 + (1.0d0 - kij(q, s))*cmp1(q)*n(q)
      end do
      t9(s) = t9(s) + t118 + (1.0d0 - kij(s, s))*cmp1(s)*n(s)
   end do

   t10 = 0.0d0
   do q = 1, nc
      t10 = t10 + n(q)
   end do

   t11 = 0.0d0
   do q = 1, nc
      t11 = t11 + t1(q)
   end do

   do s = 1, nc
      t12(s) = 0.0d0
   end do
   do s = 1, nc
      t119 = 0.0d0
      do q = 1, s - 1
         t12(q) = t12(q) + (1.0d0 - kij(q, s))*cmp1_T(s)*n(s)
         t119 = t119 + (1.0d0 - kij(q, s))*cmp1_T(q)*n(q)
      end do
      t12(s) = t12(s) + t119 + (1.0d0 - kij(s, s))*cmp1_T(s)*n(s)
   end do

   t13 = 0.0d0
   do s = 1, nc
      t13 = t13 + n(s)*t8(s)
   end do

   t14 = 0.0d0
   do s = 1, nc
      t14 = t14 + cmp1(s)*n(s)*t9(s)
   end do

   t15 = 0.0d0
   do q = 1, nc
      t15 = t15 + cmp1_T(q)*n(q)*t9(q)
   end do

   t16 = 0.0d0
   do q = 1, nc
      t16 = t16 + cmp1_TT(q)*n(q)*t9(q)
   end do

   t17 = 0.0d0
   do s = 1, nc
      t17 = t17 + cmp1_T(s)*n(s)*t12(s)
   end do

   t18 = 1d0/t10

   t19 = t11*t18

   t20 = t19 + 1.0d0

   t21 = -t11*t18 + 1.0d0

   t22 = 1d0/t20

   t23 = t10**(-2)

   t24 = V + t13*t18*t21*t22

   t25 = V + t11*t13*t23

   t26 = t20**(-2)

   t27 = t19 - t21*t22

   t28 = t10**(-3)

   t29 = 1d0/t24

   t30 = t24**(-2)

   t31 = 1d0/t13

   t32 = 1d0/t27

   t33 = 1d0/t25

   t34 = log(t25*t29)

   t35 = t27**(-2)

   t36 = t13*t21*t23*t26 + t13*t22*t23

   t37 = -t11*t13*t21*t26*t28 - t11*t13*t22*t28 + t13*t21*t22*t23

   t38 = 1d0/V

   t39 = t11*t23*t29 - t18*t21*t22*t25*t30

   t40 = -t13*t18*t38 + 1.0d0

   t41 = t13*t23*t29 + t25*t30*t36

   t42 = -2*t11*t13*t28*t29 + t25*t30*t37

   t43 = t13**(-2)

   t44 = -t18*t21*t26 - t18*t22 - t18

   t45 = t11*t21*t23*t26 + t11*t22*t23 + t11*t23

   t46 = t25**(-2)

   t47 = t24**(-3)

   t48 = t10**(-4)

   t49 = 1d0/t40

   t50 = t20**(-3)

   t51 = -t25*t30 + t29

   t52 = t40**(-2)

   t53 = V**(-2)

   t54 = t19 - 1.0d0

   t55 = t27**(-3)

   t56 = log(t40)

   t57 = t11**2

   t58 = -2*t11*t13*t21*t26*t28 - 2*t11*t13*t22*t28 + 2*t13*t21*t22*t23

   t59 = -t10*t24*t31*t32*t33*t42 - t10*t31*t34*t35*t45 - t31*t32*t34

   t60 = t11*t13*t21*t26*t28 + t11*t13*t22*t28 - t13*t21*t22*t23

   t61 = -t10*t24*t31*t32*t33*t41 - t10*t31*t34*t35*t44

   t62 = -t10*t24*t31*t32*t33*t39 + t10*t32*t34*t43

   t63 = 2*t13*t21*t23*t26 + 2*t13*t22*t23

   t64 = -t13*t21*t23*t26 - t13*t22*t23

   t65 = 1d0/(V - t13*t18*t22*t54)

   t66 = t21*t23*t26

   t67 = V**(-3)

   t68 = t22*t23

   t69 = t13**2

   t70 = t25*t30*(2*t11*t13*t21*t48*t50 + 2*t11*t13*t26*t48 - 2*t13*t21*t26*t28 - &
      2*t13*t22*t28)

   t71 = -t10*t14*t31*t34*t35*(-2*t11*t21*t28*t50 - 2*t11*t26*t28 + t23 + t66 + &
      t68)

   t72 = R*T*t38*t49 - t10*t14*t24*t31*t32*t33*t39 + t10*t14*t32*t34*t43

   t73 = 2*t11*t21*t23*t26 + 2*t11*t22*t23 + 2*t11*t23

   t74 = -2*t18*t21*t26 - 2*t18*t22 - 2*t18

   t75 = -t10*t14*t24*t31*t33*t35*t42*t44

   t76 = -t10*t14*t24*t31*t33*t35*t41*t45

   t77 = -t10*t14*t24*t31*t33*t35*t39*t45

   t78 = -t10*t14*t24*t31*t33*t35*t39*t44

   t79 = t10*t14*t24*t32*t33*t42*t43

   t80 = t10*t14*t24*t32*t33*t41*t43

   t81 = -t14*t24*t31*t32*t33*t41

   t82 = -t14*t24*t31*t32*t33*t39

   t83 = 2*t11*t13*t21*t22*t30*t48

   t84 = -R*T*t13*t23*t52*t53

   t85 = -2*t11*t13*t28*t30*t36

   t86 = t10*t14*t34*t35*t43*t45

   t87 = t10*t14*t34*t35*t43*t44

   t88 = -t14*t31*t34*t35*t44

   t89 = -t13*t21*t22*t28*t30

   t90 = -2*t13*t28*t29

   t91 = -2*t11*t28*t29

   t92 = 1d0/(t19 + t22*t54)

   t93 = t14*t32*t34*t43

   t94 = t13*t23*t30*t37

   t95 = t11*t23*t30*t37

   t96 = t11*t23*t30*t36

   t97 = -t25*t65 + 1

   t98 = t10**(-5)

   t99 = t23*t29

   t100 = -t10*t14*t24*t31*t32*t33*t41 - t10*t14*t31*t34*t35*t44

   t101 = t10*t31*t32*t34

   t102 = -R*T*t13*t18*t38*t49 - R*T*t56 - t10*t14*t24*t31*t32*t33*t42 - t10*t14* &
      t31*t34*t35*t45 - t14*t31*t32*t34

   t103 = -2*t10*t14*t24*t31*t32*t33*(-t18*t21*t22*t25*t47*t63 + t21*t23*t25*t26* &
      t30 + t22*t23*t25*t30 + t89 + t96 + t99) - 2*t10*t14*t31*t32*t33* &
      t39*t64 + 2*t14*t18*t24*t32*t39*t46 + 2*t78 + 2*t80 + 2*t87

   t104 = -2*t10*t14*t24*t31*t32*t33*(-2*t18*t21*t22*t25*t36*t47 + t25*t30*(t66 + &
      t68) + t89 + t96 + t99) + 2*t11*t14*t18*t24*t31*t32*t41*t46 - 2* &
      t14*t21*t22*t31*t32*t33*t41 + 2*t78 + 2*t80 + 2*t87

   t105 = 4*R*T*t18*t52*t53 - 4*t10*t14*t24*t31*t32*t33*(-2*t11*t21*t22*t28*t30 + &
      2*t21**2*t23*t25*t26*t47) + 8*t10*t14*t24*t32*t33*t39*t43 - 8*t10 &
      *t14*t32*t34/t13**3 + 4*t11*t14*t18*t24*t31*t32*t39*t46 - 4*t14* &
      t21*t22*t31*t32*t33*t39

   t106 = -t10*t14*t24*t31*t32*t33*(2*t13*t23*t30*t36 + t25*t30*(-2*t13*t21*t28* &
      t50 - 2*t13*t26*t28) + t25*t36*t47*t63) - 2*t10*t14*t24*t31*t33* &
      t35*t41*t44 - t10*t14*t31*t32*t33*t41*t64 - t10*t14*t31*t34*t35*( &
      2*t21*t23*t50 + 2*t23*t26) - t10*t14*t31*t34*t44*t55*t74 + t14* &
      t18*t24*t32*t41*t46

   t107 = R*T*t28*t52*t53*t69 - t10*t14*t24*t31*t32*t33*(-4*t11*t13*t28*t30*t37 + &
      6*t11*t13*t29*t48 + t25*t30*(4*t11*t13*t21*t26*t48 + 4*t11*t13* &
      t22*t48 - 2*t13*t21*t22*t28 - 2*t13*t21*t50*t57*t98 - 2*t13*t26* &
      t57*t98) + t25*t37*t47*t58) - 2*t10*t14*t24*t31*t33*t35*t42*t45 - &
      t10*t14*t31*t32*t33*t42*t60 - t10*t14*t31*t34*t35*(-2*t11*t21*t26 &
      *t28 - 2*t11*t22*t28 - 2*t11*t28 + 2*t21*t48*t50*t57 + 2*t26*t48* &
      t57) - t10*t14*t31*t34*t45*t55*t73 - 2*t11*t14*t23*t24*t32*t42* &
      t46 - 2*t14*t24*t31*t32*t33*t42 - 2*t14*t31*t34*t35*t45

   do u = 1, nc
      t108(u) = 2*t59*cmp1(u)*t9(u) + (-t10*t14*t24*t31*t32*t33*(t25*t36*t47*t58 + t70 + &
      t85 + t90 + t94) - t10*t14*t31*t32*t33*t41*t60 - t10*t14*t31*t34* &
      t44*t55*t73 - 2*t11*t14*t23*t24*t32*t41*t46 + t71 + t75 + t76 + &
      t81 + t88)*delta1(u) + (-2*t10*t14*t24*t31*t32*t33*(-t11*t21*t25* &
      t26*t28*t30 - t11*t22*t25*t28*t30 - t18*t21*t22*t25*t47*t58 + t21 &
      *t22*t23*t25*t30 + t83 + t91 + t95) - 2*t10*t14*t31*t32*t33*t39* &
      t60 - 4*t11*t14*t23*t24*t32*t39*t46 + 2*t77 + 2*t79 + 2*t82 + 2* &
      t84 + 2*t86 + 2*t93)*t8(u)
   end do

   do u = 1, nc
      t109(u) = 2*t59*cmp1(u)*t9(u) + (-t10*t14*t24*t31*t32*t33*(t25*t37*t47*t63 + t70 + &
      t85 + t90 + t94) - t10*t14*t31*t32*t33*t42*t64 - t10*t14*t31*t34* &
      t45*t55*t74 + t14*t18*t24*t32*t42*t46 + t71 + t75 + t76 + t81 + &
      t88)*delta1(u) + (-2*t10*t14*t24*t31*t32*t33*(-2*t18*t21*t22*t25* &
      t37*t47 + t25*t30*(-t11*t21*t26*t28 - t11*t22*t28 + t21*t22*t23) &
      + t83 + t91 + t95) + 2*t11*t14*t18*t24*t31*t32*t42*t46 - 2*t14* &
      t21*t22*t31*t32*t33*t42 + 2*t77 + 2*t79 + 2*t82 + 2*t84 + 2*t86 + &
      2*t93)*t8(u)
   end do

   t110 = 2*R*t38*t49 + 4*t15*t62

   t111 = t15*t61

   t112 = -R*t13*t18*t38*t49 - R*t56 + 2*t15*t59

   t113 = -t10*t14*t24*t31*t32*t33*(-t13*t23*t30 - 2*t25*t36*t47 + t30*t36) + t10* &
      t14*t24*t31*t32*t41*t46 - t10*t14*t24*t31*t33*t35*t44*t51 - t10* &
      t14*t31*t32*t33*t41

   t114 = -2*R*T*t13*t18*t52*t67 - 2*R*T*t49*t53 - 2*t10*t14*t24*t31*t32*t33*(-t11 &
      *t23*t30 + 2*t18*t21*t22*t25*t47 - t18*t21*t22*t30) + 2*t10*t14* &
      t24*t31*t32*t39*t46 + 2*t10*t14*t24*t32*t33*t43*t51 - 2*t10*t14* &
      t31*t32*t33*t39

   t115 = t10*t24*t31*t32*t33*t51

   t116 = R*T*t23*t52*t67*t69 - t10*t14*t24*t31*t32*t33*(2*t11*t13*t28*t30 - 2*t25 &
      *t37*t47 + t30*t37) + t10*t14*t24*t31*t32*t42*t46 - t10*t14*t24* &
      t31*t33*t35*t45*t51 - t10*t14*t31*t32*t33*t42 - t14*t24*t31*t32* &
      t33*t51

   res = -R*T*t10*t56 - t10*t14*t31*t32*t34
   f = res

   res = -R*t10*t56 - 2*t10*t15*t31*t32*t34
   dfdT = res

   res = -R*T*t13*t49*t53 - t10*t14*t24*t31*t32*t33*t51
   dfdV = res

   do i = 1, nc
      res = t100*delta1(i) - 2*t101*cmp1(i)*t9(i) + t102 + 2*t72*t8(i)
      dfdn(i) = res
   end do

   res = -t10*t31*t32*t34*(2*t16 + 2*t17)
   dfdT2 = res

   res = 2*R*T*t13*t49*t67 + R*T*t18*t52*t69/V**4 + t10*t14*t31*t33*t65*t92 &
         *t97 + t10*t14*t31*t46*t92*t97
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = -2*t101*(1.0d0 - kij(i, j))*cmp1(i)*cmp1(j) + t103*delta1(j)*t8(i &
               ) + t104*delta1(i)*t8(j) + t105*t8(i)*t8(j) + t106*delta1(i)* &
               delta1(j) + t107 + 2*t61*cmp1(i)*delta1(j)*t9(i) + 2*t61*cmp1(j)* &
               delta1(i)*t9(j) + 4*t62*cmp1(i)*t8(j)*t9(i) + 4*t62*cmp1(j)*t8(i) &
               *t9(j) + 2*t72*mat2(i, j) + t108(i) + t109(j)
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t13*t49*t53 - 2*t10*t15*t24*t31*t32*t33*t51
   dfdTdV = res

   do i = 1, nc
      res = -t101*(2*cmp1(i)*t12(i) + 2*cmp1_T(i)*t9(i)) + t110*t8(i) + 2*t111 &
            *delta1(i) + t112
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = t113*delta1(i) + t114*t8(i) - 2*t115*cmp1(i)*t9(i) + t116
      dfdndV(i) = res
   end do

end subroutine rkpr_c6m_w6
