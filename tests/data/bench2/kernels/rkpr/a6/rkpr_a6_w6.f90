subroutine rkpr_a6_w6(nc, V, T, Tc, ac, b, delta1, k, kij, lij, n, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, &
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
   real(8) :: res, t100, t101, t35, t36, t37, t38, t39, t40, t41, t44, t45, t50, t51, t52, t53, t54, t57, t58, t59, t66
   real(8) :: t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80, t81, t82, t83, t84, t85, t86, t87, t88
   real(8) :: t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99
   real(8) :: t1(nc), t10(nc), t13(nc), t14(nc), t15(nc), t16(nc), t17(nc), t18(nc), t20(nc), t21(nc), t22(nc), t25(nc)
   real(8) :: t26(nc), t28(nc), t29(nc), t30(nc), t31(nc), t32(nc), t33(nc), t4(nc), t42(nc), t43(nc), t46(nc), t47(nc)
   real(8) :: t48(nc), t49(nc), t5(nc), t55(nc), t56(nc), t6(nc), t60(nc), t61(nc), t62(nc), t63(nc), t64(nc), t65(nc)
   real(8) :: t7(nc), t8(nc), t9(nc)
   real(8) :: t11(nc, nc), t12(nc, nc), t19(nc, nc), t2(nc, nc), t23(nc, nc), t24(nc, nc), t27(nc, nc), t3(nc, nc)
   real(8) :: t34(nc, nc)
   integer :: i, j, l, m, p, q, s, u

   do p = 1, nc
      t1(p) = 0.5d0*b(p)
   end do

   do p = 1, nc
      do u = p, nc
         t2(p, u) = 1.0d0 - lij(p, u)
         t2(u, p) = t2(p, u)
      end do
   end do

   do p = 1, nc
      do u = p, nc
         t3(p, u) = t1(p) + t1(u)
         t3(u, p) = t3(p, u)
      end do
   end do

   do p = 1, nc
      t4(p) = n(p)**2
   end do

   do p = 1, nc
      t5(p) = 1.0d0 - lij(p, p)
   end do

   do p = 1, nc
      t6(p) = 1d0/Tc(p)
   end do

   do p = 1, nc
      t7(p) = T*t6(p) + 2.0d0
   end do

   do p = 1, nc
      t8(p) = 1d0/t7(p)
   end do

   do p = 1, nc
      t9(p) = 3.0d0*t8(p)
   end do

   do p = 1, nc
      t10(p) = t9(p)**k(p)
   end do

   do p = 1, nc
      do u = p, nc
         t11(p, u) = sqrt(ac(p)*ac(u)*t10(p)*t10(u))
         t11(u, p) = t11(p, u)
      end do
   end do

   do p = 1, nc
      do u = p, nc
         t12(p, u) = 1.0d0 - kij(p, u)
         t12(u, p) = t12(p, u)
      end do
   end do

   do p = 1, nc
      t13(p) = 0.333333333333333d0*T*t6(p) + 0.666666666666667d0
   end do

   do p = 1, nc
      t14(p) = t9(p)**(2*k(p))
   end do

   do p = 1, nc
      t15(p) = ac(p)**2
   end do

   do p = 1, nc
      t16(p) = sqrt(t14(p)*t15(p))
   end do

   do p = 1, nc
      t17(p) = 1.0d0 - kij(p, p)
   end do

   do p = 1, nc
      t18(p) = t7(p)**(-2)
   end do

   do p = 1, nc
      do u = 1, nc
         t19(p, u) = n(p)*n(u)*t11(p, u)*t12(p, u)
      end do
   end do

   do p = 1, nc
      t20(p) = t16(p)*t17(p)*t4(p)
   end do

   do p = 1, nc
      t21(p) = t9(p)**(-k(p))
   end do

   do p = 1, nc
      t22(p) = 1d0/ac(p)
   end do

   do p = 1, nc
      do u = 1, nc
         t23(p, u) = n(u)*t11(p, u)*t12(p, u)
      end do
   end do

   do p = 1, nc
      do u = p, nc
         t24(p, u) = -1.5d0*ac(p)*ac(u)*k(p)*t10(p)*t10(u)*t13(p)*t18(p)*t6(p) - 1.5d0*ac(p)* &
      ac(u)*k(u)*t10(p)*t10(u)*t13(u)*t18(u)*t6(u)
         t24(u, p) = t24(p, u)
      end do
   end do

   do p = 1, nc
      t25(p) = Tc(p)**(-2)
   end do

   do p = 1, nc
      t26(p) = -3.0d0*k(p)*t13(p)*t16(p)*t17(p)*t18(p)*t4(p)*t6(p)
   end do

   do p = 1, nc
      do u = 1, nc
         t27(p, u) = n(p)*n(u)*t11(p, u)*t12(p, u)*t21(p)*t21(u)*t22(p)*t22(u)*t24(p, u)
      end do
   end do

   do p = 1, nc
      t28(p) = t7(p)**(-4)
   end do

   do p = 1, nc
      t29(p) = k(p)**2
   end do

   do p = 1, nc
      t30(p) = t13(p)**2
   end do

   do p = 1, nc
      t31(p) = t9(p)**(-2*k(p))
   end do

   do p = 1, nc
      t32(p) = ac(p)**(-2)
   end do

   do p = 1, nc
      t33(p) = t7(p)**(-3)
   end do

   do p = 1, nc
      do u = 1, nc
         t34(p, u) = n(u)*t11(p, u)*t12(p, u)*t21(p)*t21(u)*t22(p)*t22(u)*t24(p, u)
      end do
   end do

   t35 = 0.0d0
   do l = 1, nc
      t35 = t35 + n(l)
   end do

   t36 = 0.0d0
   do m = 1, nc
      t36 = t36 + n(m)
   end do

   t37 = 0.0d0
   do q = 1, nc
      t37 = t37 + n(q)
   end do

   t38 = 0.0d0
   do q = 1, nc
      t38 = t38 + t20(q)
   end do

   t39 = 0.0d0
   do q = 1, nc
      do s = q + 1, nc
         t39 = t39 + t19(q, s)
      end do
   end do

   t40 = 0.0d0
   do q = 1, nc
      t40 = t40 + t26(q)
   end do

   t41 = 0.0d0
   do q = 1, nc
      do s = q + 1, nc
         t41 = t41 + t27(q, s)
      end do
   end do

   do i = 1, nc
      t42(i) = 0.0d0
      do q = 1, nc
         t42(i) = t42(i) + t23(i, q)
      end do
   end do

   do i = 1, nc
      t43(i) = 0.0d0
      do s = 1, nc
         t43(i) = t43(i) + t23(i, s)
      end do
   end do

   t44 = 0.0d0
   do q = 1, nc
      t44 = t44 + (6.0d0*k(q)*t13(q)*t14(q)*t15(q)*t25(q)*t33(q) - 1.0d0*k(q)*t14(q)*t15(q &
      )*t18(q)*t25(q) + 18.0d0*t14(q)*t15(q)*t25(q)*t28(q)*t29(q)*t30(q &
      ))*t16(q)*t17(q)*t31(q)*t32(q)*t4(q) - 9.0d0*t16(q)*t17(q)*t25(q) &
      *t28(q)*t29(q)*t30(q)*t4(q)
   end do

   t45 = 0.0d0
   do q = 1, nc
      do s = q + 1, nc
         t45 = t45 + (9.0d0*ac(q)*ac(s)*k(q)*k(s)*t10(q)*t10(s)*t13(q)*t13(s)*t18(q)*t18(s)* &
      t6(q)*t6(s) + 3.0d0*ac(q)*ac(s)*k(q)*t10(q)*t10(s)*t13(q)*t25(q)* &
      t33(q) - 0.5d0*ac(q)*ac(s)*k(q)*t10(q)*t10(s)*t18(q)*t25(q) + &
      3.0d0*ac(q)*ac(s)*k(s)*t10(q)*t10(s)*t13(s)*t25(s)*t33(s) - 0.5d0 &
      *ac(q)*ac(s)*k(s)*t10(q)*t10(s)*t18(s)*t25(s) + 4.5d0*ac(q)*ac(s) &
      *t10(q)*t10(s)*t25(q)*t28(q)*t29(q)*t30(q) + 4.5d0*ac(q)*ac(s)* &
      t10(q)*t10(s)*t25(s)*t28(s)*t29(s)*t30(s))*n(q)*n(s)*t11(q, s)* &
      t12(q, s)*t21(q)*t21(s)*t22(q)*t22(s) + 3.0d0*k(q)*n(q)*n(s)*t11( &
      q, s)*t12(q, s)*t13(q)*t18(q)*t21(q)*t21(s)*t22(q)*t22(s)*t24(q, &
      s)*t6(q) + 3.0d0*k(s)*n(q)*n(s)*t11(q, s)*t12(q, s)*t13(s)*t18(s) &
      *t21(q)*t21(s)*t22(q)*t22(s)*t24(q, s)*t6(s) + n(q)*n(s)*t11(q, s &
      )*t12(q, s)*t24(q, s)**2*t31(q)*t31(s)*t32(q)*t32(s)
      end do
   end do

   do j = 1, nc
      t46(j) = 0.0d0
      do q = 1, nc
         t46(j) = t46(j) + t23(j, q)
      end do
   end do

   do j = 1, nc
      t47(j) = 0.0d0
      do s = 1, nc
         t47(j) = t47(j) + t23(j, s)
      end do
   end do

   do i = 1, nc
      t48(i) = 0.0d0
      do q = 1, nc
         t48(i) = t48(i) + t34(i, q)
      end do
   end do

   do i = 1, nc
      t49(i) = 0.0d0
      do s = 1, nc
         t49(i) = t49(i) + t34(i, s)
      end do
   end do

   t50 = 0.0d0
   do q = 1, nc
      t50 = t50 + delta1(q)*n(q)/t36
   end do

   t51 = 0.0d0
   do q = 1, nc
      t51 = t51 + 1.0d0*b(q)*t4(q)*t5(q)/t35
   end do

   t52 = 0.0d0
   do q = 1, nc
      do s = q + 1, nc
         t52 = t52 + n(q)*n(s)*t2(q, s)*t3(q, s)/t35
      end do
   end do

   t53 = 0.0d0
   do q = 1, nc
      t53 = t53 + -delta1(q)*n(q)/t36**2
   end do

   t54 = 0.0d0
   do q = 1, nc
      t54 = t54 + -1.0d0*b(q)*t4(q)*t5(q)/t35**2
   end do

   do i = 1, nc
      t55(i) = 0.0d0
      do q = 1, nc
         t55(i) = t55(i) + n(q)*t2(i, q)*t3(i, q)/t35
      end do
   end do

   do i = 1, nc
      t56(i) = 0.0d0
      do s = 1, nc
         t56(i) = t56(i) + n(s)*t2(i, s)*t3(i, s)/t35
      end do
   end do

   t57 = 0.0d0
   do q = 1, nc
      do s = q + 1, nc
         t57 = t57 + -n(q)*n(s)*t2(q, s)*t3(q, s)/t35**2
      end do
   end do

   t58 = 0.0d0
   do q = 1, nc
      t58 = t58 + 2*delta1(q)*n(q)/t36**3
   end do

   t59 = 0.0d0
   do q = 1, nc
      t59 = t59 + 2.0d0*b(q)*t4(q)*t5(q)/t35**3
   end do

   do i = 1, nc
      t60(i) = 0.0d0
      do q = 1, nc
         t60(i) = t60(i) + -n(q)*t2(i, q)*t3(i, q)/t35**2
      end do
   end do

   do j = 1, nc
      t61(j) = 0.0d0
      do q = 1, nc
         t61(j) = t61(j) + -n(q)*t2(j, q)*t3(j, q)/t35**2
      end do
   end do

   do i = 1, nc
      t62(i) = 0.0d0
      do s = 1, nc
         t62(i) = t62(i) + -n(s)*t2(i, s)*t3(i, s)/t35**2
      end do
   end do

   do j = 1, nc
      t63(j) = 0.0d0
      do s = 1, nc
         t63(j) = t63(j) + -n(s)*t2(j, s)*t3(j, s)/t35**2
      end do
   end do

   do j = 1, nc
      t64(j) = 0.0d0
      do q = 1, nc
         t64(j) = t64(j) + n(q)*t2(j, q)*t3(j, q)/t35
      end do
   end do

   do j = 1, nc
      t65(j) = 0.0d0
      do s = 1, nc
         t65(j) = t65(j) + n(s)*t2(j, s)*t3(j, s)/t35
      end do
   end do

   t66 = 0.0d0
   do q = 1, nc
      do s = q + 1, nc
         t66 = t66 + 2*n(q)*n(s)*t2(q, s)*t3(q, s)/t35**3
      end do
   end do

   t67 = t51 + 2*t52

   t68 = t50 + 1.0d0

   t69 = 1.0d0 - t50

   t70 = 1d0/t68

   t71 = 1d0/t36

   t72 = V + t67*t69*t70

   t73 = V + t50*t67

   t74 = 2*t57

   t75 = t50 - t69*t70

   t76 = 1d0/t72

   t77 = t68**(-2)

   t78 = 1d0/t75

   t79 = 1d0/t67

   t80 = t38 + 2*t39

   t81 = t72**(-2)

   t82 = 1d0/V

   t83 = log(t73*t76)

   t84 = 1d0/t73

   t85 = -t67*t82 + 1.0d0

   t86 = t36**(-2)

   t87 = t75**(-2)

   t88 = t67**(-2)

   t89 = 1d0/t85

   t90 = -t53

   t91 = -t73*t81 + t76

   t92 = t40 + 2*t41

   t93 = V**(-2)

   t94 = log(t85)

   t95 = 2*t66

   t96 = 1d0/t35

   t97 = t85**(-2)

   t98 = t73**(-2)

   t99 = t72**(-3)

   t100 = t68**(-3)

   t101 = V**(-3)

   res = -R*T*t37*t94 - t78*t79*t80*t83
   f = res

   res = -R*t37*t94 - t78*t79*t83*t92
   dfdT = res

   res = -R*T*t37*t67*t89*t93 - t72*t78*t79*t80*t84*t91
   dfdV = res

   do i = 1, nc
      res = R*T*t37*t82*t89*(t54 + t74 + t55(i) + t56(i)) - R*T*t94 - t72*t78* &
            t79*t80*t84*(t73*t81*(t67*t69*t77*(t53 + t71*delta1(i)) + t67*t70 &
            *(t53 + t71*delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i))) + &
            t76*(t50*(t54 + t74 + t55(i) + t56(i)) + t67*(t53 + t71*delta1(i &
            )))) - t78*t79*t83*(t42(i) + t43(i)) + t78*t80*t83*t88*(t54 + t74 &
            + t55(i) + t56(i)) - t79*t80*t83*t87*(-t69*t77*(t53 + t71*delta1( &
            i)) - t70*(t53 + t71*delta1(i)) - t71*delta1(i) + t90)
      dfdn(i) = res
   end do

   res = -t78*t79*t83*(t44 + 2*t45)
   dfdT2 = res

   res = 2*R*T*t101*t37*t67*t89 + R*T*t37*t67**2*t97/V**4 - t72*t78*t79*t80 &
         *t84*(2*t73*t99 - 2*t81) + t72*t78*t79*t80*t91*t98 - t78*t79*t80* &
         t84*t91
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = R*T*t37*t82*t89*(t59 + t95 + 2*t96*t2(i, j)*t3(i, j) + t60(i) + &
               t61(j) + t62(i) + t63(j)) + R*T*t37*t93*t97*(t54 + t74 + t55(i) + &
               t56(i))*(t54 + t74 + t64(j) + t65(j)) + R*T*t82*t89*(t54 + t74 + &
               t55(i) + t56(i)) + R*T*t82*t89*(t54 + t74 + t64(j) + t65(j)) - &
               t72*t78*t79*t80*t84*(t73*t81*(-2*t100*t67*t69*(t53 + t71*delta1(i &
               ))*(t53 + t71*delta1(j)) + t67*t69*t77*(t58 - t86*delta1(i) - t86 &
               *delta1(j)) + t67*t70*(t58 - t86*delta1(i) - t86*delta1(j)) - 2* &
               t67*t77*(t53 + t71*delta1(i))*(t53 + t71*delta1(j)) - t69*t70*( &
               t59 + t95 + 2*t96*t2(i, j)*t3(i, j) + t60(i) + t61(j) + t62(i) + &
               t63(j)) + t69*t77*(t53 + t71*delta1(i))*(t54 + t74 + t64(j) + t65 &
               (j)) + t69*t77*(t53 + t71*delta1(j))*(t54 + t74 + t55(i) + t56(i &
               )) + t70*(t53 + t71*delta1(i))*(t54 + t74 + t64(j) + t65(j)) + &
               t70*(t53 + t71*delta1(j))*(t54 + t74 + t55(i) + t56(i))) + t73* &
               t99*(t67*t69*t77*(t53 + t71*delta1(i)) + t67*t70*(t53 + t71* &
               delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i)))*(2*t67*t69* &
               t77*(t53 + t71*delta1(j)) + 2*t67*t70*(t53 + t71*delta1(j)) - 2* &
               t69*t70*(t54 + t74 + t64(j) + t65(j))) + t76*(t50*(t59 + t95 + 2* &
               t96*t2(i, j)*t3(i, j) + t60(i) + t61(j) + t62(i) + t63(j)) + t67* &
               (t58 - t86*delta1(i) - t86*delta1(j)) + (t53 + t71*delta1(i))*( &
               t54 + t74 + t64(j) + t65(j)) + (t53 + t71*delta1(j))*(t54 + t74 + &
               t55(i) + t56(i))) + t81*(t50*(t54 + t74 + t55(i) + t56(i)) + t67* &
               (t53 + t71*delta1(i)))*(t67*t69*t77*(t53 + t71*delta1(j)) + t67* &
               t70*(t53 + t71*delta1(j)) - t69*t70*(t54 + t74 + t64(j) + t65(j &
               ))) + t81*(t50*(t54 + t74 + t64(j) + t65(j)) + t67*(t53 + t71* &
               delta1(j)))*(t67*t69*t77*(t53 + t71*delta1(i)) + t67*t70*(t53 + &
               t71*delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i)))) - t72* &
               t78*t79*t80*t98*(-t50*(t54 + t74 + t64(j) + t65(j)) - t67*(t53 + &
               t71*delta1(j)))*(t73*t81*(t67*t69*t77*(t53 + t71*delta1(i)) + t67 &
               *t70*(t53 + t71*delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i &
               ))) + t76*(t50*(t54 + t74 + t55(i) + t56(i)) + t67*(t53 + t71* &
               delta1(i)))) - t72*t78*t79*t84*(t73*t81*(t67*t69*t77*(t53 + t71* &
               delta1(i)) + t67*t70*(t53 + t71*delta1(i)) - t69*t70*(t54 + t74 + &
               t55(i) + t56(i))) + t76*(t50*(t54 + t74 + t55(i) + t56(i)) + t67* &
               (t53 + t71*delta1(i))))*(t46(j) + t47(j)) - t72*t78*t79*t84*(t73* &
               t81*(t67*t69*t77*(t53 + t71*delta1(j)) + t67*t70*(t53 + t71* &
               delta1(j)) - t69*t70*(t54 + t74 + t64(j) + t65(j))) + t76*(t50*( &
               t54 + t74 + t64(j) + t65(j)) + t67*(t53 + t71*delta1(j))))*(t42(i &
               ) + t43(i)) + t72*t78*t80*t84*t88*(t73*t81*(t67*t69*t77*(t53 + &
               t71*delta1(i)) + t67*t70*(t53 + t71*delta1(i)) - t69*t70*(t54 + &
               t74 + t55(i) + t56(i))) + t76*(t50*(t54 + t74 + t55(i) + t56(i)) &
               + t67*(t53 + t71*delta1(i))))*(t54 + t74 + t64(j) + t65(j)) + t72 &
               *t78*t80*t84*t88*(t73*t81*(t67*t69*t77*(t53 + t71*delta1(j)) + &
               t67*t70*(t53 + t71*delta1(j)) - t69*t70*(t54 + t74 + t64(j) + t65 &
               (j))) + t76*(t50*(t54 + t74 + t64(j) + t65(j)) + t67*(t53 + t71* &
               delta1(j))))*(t54 + t74 + t55(i) + t56(i)) - t72*t79*t80*t84*t87* &
               (t73*t81*(t67*t69*t77*(t53 + t71*delta1(i)) + t67*t70*(t53 + t71* &
               delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i))) + t76*(t50*( &
               t54 + t74 + t55(i) + t56(i)) + t67*(t53 + t71*delta1(i))))*(-t69* &
               t77*(t53 + t71*delta1(j)) - t70*(t53 + t71*delta1(j)) - t71* &
               delta1(j) + t90) - t72*t79*t80*t84*t87*(t73*t81*(t67*t69*t77*(t53 &
               + t71*delta1(j)) + t67*t70*(t53 + t71*delta1(j)) - t69*t70*(t54 + &
               t74 + t64(j) + t65(j))) + t76*(t50*(t54 + t74 + t64(j) + t65(j)) &
               + t67*(t53 + t71*delta1(j))))*(-t69*t77*(t53 + t71*delta1(i)) - &
               t70*(t53 + t71*delta1(i)) - t71*delta1(i) + t90) - t78*t79*t80* &
               t84*(t73*t81*(t67*t69*t77*(t53 + t71*delta1(i)) + t67*t70*(t53 + &
               t71*delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i))) + t76*( &
               t50*(t54 + t74 + t55(i) + t56(i)) + t67*(t53 + t71*delta1(i))))*( &
               -t67*t69*t77*(t53 + t71*delta1(j)) - t67*t70*(t53 + t71*delta1(j &
               )) + t69*t70*(t54 + t74 + t64(j) + t65(j))) - 2*t78*t79*t83*t11(i &
               , j)*t12(i, j) + t78*t80*t83*t88*(t59 + t95 + 2*t96*t2(i, j)*t3(i &
               , j) + t60(i) + t61(j) + t62(i) + t63(j)) + t78*t83*t88*(t42(i) + &
               t43(i))*(t54 + t74 + t64(j) + t65(j)) + t78*t83*t88*(t46(j) + t47 &
               (j))*(t54 + t74 + t55(i) + t56(i)) - t79*t80*t83*t87*(2*t100*t69* &
               (t53 + t71*delta1(i))*(t53 + t71*delta1(j)) - t58 - t69*t77*(t58 &
               - t86*delta1(i) - t86*delta1(j)) - t70*(t58 - t86*delta1(i) - t86 &
               *delta1(j)) + 2*t77*(t53 + t71*delta1(i))*(t53 + t71*delta1(j)) + &
               t86*delta1(i) + t86*delta1(j)) - t79*t83*t87*(t42(i) + t43(i))*( &
               -t69*t77*(t53 + t71*delta1(j)) - t70*(t53 + t71*delta1(j)) - t71* &
               delta1(j) + t90) - t79*t83*t87*(t46(j) + t47(j))*(-t69*t77*(t53 + &
               t71*delta1(i)) - t70*(t53 + t71*delta1(i)) - t71*delta1(i) + t90 &
               ) + t80*t83*t87*t88*(t54 + t74 + t55(i) + t56(i))*(-t69*t77*(t53 &
               + t71*delta1(j)) - t70*(t53 + t71*delta1(j)) - t71*delta1(j) + &
               t90) + t80*t83*t87*t88*(t54 + t74 + t64(j) + t65(j))*(-t69*t77*( &
               t53 + t71*delta1(i)) - t70*(t53 + t71*delta1(i)) - t71*delta1(i) &
               + t90) - t79*t80*t83*(-2*t53 - 2*t69*t77*(t53 + t71*delta1(j)) - &
               2*t70*(t53 + t71*delta1(j)) - 2*t71*delta1(j))*(-t69*t77*(t53 + &
               t71*delta1(i)) - t70*(t53 + t71*delta1(i)) - t71*delta1(i) + t90) &
               /t75**3 - 2*t78*t80*t83*(t54 + t74 + t55(i) + t56(i))*(t54 + t74 &
               + t64(j) + t65(j))/t67**3
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t37*t67*t89*t93 - t72*t78*t79*t84*t91*t92
   dfdTdV = res

   do i = 1, nc
      res = R*t37*t82*t89*(t54 + t74 + t55(i) + t56(i)) - R*t94 - t72*t78*t79* &
            t84*t92*(t73*t81*(t67*t69*t77*(t53 + t71*delta1(i)) + t67*t70*( &
            t53 + t71*delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i))) + &
            t76*(t50*(t54 + t74 + t55(i) + t56(i)) + t67*(t53 + t71*delta1(i &
            )))) - t78*t79*t83*(t48(i) + t49(i)) + t78*t83*t88*t92*(t54 + t74 &
            + t55(i) + t56(i)) - t79*t83*t87*t92*(-t69*t77*(t53 + t71*delta1( &
            i)) - t70*(t53 + t71*delta1(i)) - t71*delta1(i) + t90)
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = -R*T*t101*t37*t67*t97*(t54 + t74 + t55(i) + t56(i)) - R*T*t37*t89* &
            t93*(t54 + t74 + t55(i) + t56(i)) - R*T*t67*t89*t93 - t72*t78*t79 &
            *t80*t84*(-t73*t99*(2*t67*t69*t77*(t53 + t71*delta1(i)) + 2*t67* &
            t70*(t53 + t71*delta1(i)) - 2*t69*t70*(t54 + t74 + t55(i) + t56(i &
            ))) - t81*(t50*(t54 + t74 + t55(i) + t56(i)) + t67*(t53 + t71* &
            delta1(i))) + t81*(t67*t69*t77*(t53 + t71*delta1(i)) + t67*t70*( &
            t53 + t71*delta1(i)) - t69*t70*(t54 + t74 + t55(i) + t56(i)))) - &
            t72*t78*t79*t80*t91*t98*(-t50*(t54 + t74 + t55(i) + t56(i)) - t67 &
            *(t53 + t71*delta1(i))) - t72*t78*t79*t84*t91*(t42(i) + t43(i)) + &
            t72*t78*t80*t84*t88*t91*(t54 + t74 + t55(i) + t56(i)) - t72*t79* &
            t80*t84*t87*t91*(-t69*t77*(t53 + t71*delta1(i)) - t70*(t53 + t71* &
            delta1(i)) - t71*delta1(i) + t90) - t78*t79*t80*t84*t91*(-t67*t69 &
            *t77*(t53 + t71*delta1(i)) - t67*t70*(t53 + t71*delta1(i)) + t69* &
            t70*(t54 + t74 + t55(i) + t56(i)))
      dfdndV(i) = res
   end do

end subroutine rkpr_a6_w6
