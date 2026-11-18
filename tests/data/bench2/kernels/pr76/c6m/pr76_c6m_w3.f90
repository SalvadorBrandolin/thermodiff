subroutine pr76_c6m_w3(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdn)
   use yaeos__constants, only: R
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), intent(in) :: Pc(:)
   real(8), intent(in) :: Tc(:)
   real(8), intent(in) :: kij(:, :)
   real(8), intent(in) :: lij(:, :)
   real(8), intent(in) :: n(:)
   real(8), intent(in) :: w(:)
   real(8), intent(out) :: f
   real(8), intent(out) :: dfdn(:)
   real(8) :: res, t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t2, t20, t21, t22, t23, t25, t26, t27, t28, t29
   real(8) :: t3, t4, t7, t8, t9
   real(8) :: cmp0(nc), cmp1(nc), t24(nc), t5(nc), t6(nc)
   real(8) :: mat2(nc, nc)
   integer :: i, ia, ib, ic, k, l, m, p

   do ic = 1, nc
      cmp0(ic) = ((1 - sqrt(T/Tc(ic)))*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0 &
      ) + 1)**2*Tc(ic)**2/Pc(ic)
   end do

   do ic = 1, nc
      cmp1(ic) = sqrt(cmp0(ic))
   end do

   do p = 1, nc
      t24(p) = 0.038898035d0*R*Tc(p)/Pc(p)
   end do

   do ia = 1, nc
      do ib = ia, nc
         mat2(ia, ib) = (1.0d0 - lij(ia, ib))*(t24(ia) + t24(ib))
         mat2(ib, ia) = mat2(ia, ib)
      end do
   end do

   do l = 1, nc
      t5(l) = 0.0d0
   end do
   do l = 1, nc
      t28 = 0.0d0
      do k = 1, l - 1
         t5(k) = t5(k) + mat2(k, l)*n(l)
         t28 = t28 + mat2(k, l)*n(k)
      end do
      t5(l) = t5(l) + t28 + mat2(l, l)*n(l)
   end do

   do l = 1, nc
      t6(l) = 0.0d0
   end do
   do l = 1, nc
      t29 = 0.0d0
      do k = 1, l - 1
         t6(k) = t6(k) + (1.0d0 - kij(k, l))*cmp1(l)*n(l)
         t29 = t29 + (1.0d0 - kij(k, l))*cmp1(k)*n(k)
      end do
      t6(l) = t6(l) + t29 + (1.0d0 - kij(l, l))*cmp1(l)*n(l)
   end do

   t7 = 0.0d0
   do m = 1, nc
      t7 = t7 + n(m)
   end do

   t8 = 0.0d0
   do l = 1, nc
      t8 = t8 + n(l)*t5(l)
   end do

   t9 = 0.0d0
   do l = 1, nc
      t9 = t9 + cmp1(l)*n(l)*t6(l)
   end do

   t10 = 1d0/t7

   t11 = V - 0.414213562373095d0*t10*t8

   t12 = V + 2.41421356237309d0*t10*t8

   t13 = sqrt(R**4)

   t14 = 1d0/t11

   t15 = 1d0/V

   t16 = 1d0/t8

   t17 = -t10*t15*t8 + 1.0d0

   t18 = log(t12*t14)

   t19 = t7**(-2)

   t20 = t11**(-2)

   t21 = log(t17)

   t22 = 1d0/t17

   t23 = 1d0/t12

   t25 = 2*R*T*t15*t22 - 0.323314343862425d0*t11*t13*t16*t23*t7*t9*( &
      0.414213562373095d0*t10*t12*t20 + 2.41421356237309d0*t10*t14) + &
      0.323314343862425d0*t13*t18*t7*t9/t8**2

   t26 = t13*t16*t18*t7

   t27 = -R*T*t10*t15*t22*t8 - R*T*t21 - 0.161657171931213d0*t11*t13*t16*t23*t7* &
      t9*(-0.414213562373095d0*t12*t19*t20*t8 - 2.41421356237309d0*t14* &
      t19*t8) - 0.161657171931213d0*t13*t16*t18*t9

   res = -R*T*t21*t7 - 0.161657171931213d0*t13*t16*t18*t7*t9
   f = res

   do i = 1, nc
      res = t25*t5(i) - 0.323314343862425d0*t26*cmp1(i)*t6(i) + t27
      dfdn(i) = res
   end do

end subroutine pr76_c6m_w3
