subroutine pr76_c6n_w3(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdn)
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
   real(8) :: res, t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t2, t20, t21, t23, t24, t25, t5, t6, t7, t8, t9
   real(8) :: cmp0(nc), t22(nc), t3(nc), t4(nc)
   real(8) :: mat1(nc, nc), mat2(nc, nc)
   integer :: i, ia, ib, ic, k, l, m, p

   do ic = 1, nc
      cmp0(ic) = ((1 - sqrt(T/Tc(ic)))*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0 &
      ) + 1)**2*Tc(ic)**2/Pc(ic)
   end do

   t21 = R**4

   do ia = 1, nc
      do ib = ia, nc
         mat1(ia, ib) = sqrt(t21*cmp0(ia)*cmp0(ib))*(1.0d0 - kij(ia, ib))
         mat1(ib, ia) = mat1(ia, ib)
      end do
   end do

   do p = 1, nc
      t22(p) = 0.038898035d0*R*Tc(p)/Pc(p)
   end do

   do ia = 1, nc
      do ib = ia, nc
         mat2(ia, ib) = (1.0d0 - lij(ia, ib))*(t22(ia) + t22(ib))
         mat2(ib, ia) = mat2(ia, ib)
      end do
   end do

   do l = 1, nc
      t3(l) = 0.0d0
      do k = 1, nc
         t3(l) = t3(l) + mat1(k, l)*n(k)
      end do
   end do

   do l = 1, nc
      t4(l) = 0.0d0
      do k = 1, nc
         t4(l) = t4(l) + mat2(k, l)*n(k)
      end do
   end do

   t5 = 0.0d0
   do m = 1, nc
      t5 = t5 + n(m)
   end do

   t6 = 0.0d0
   do l = 1, nc
      t6 = t6 + n(l)*t3(l)
   end do

   t7 = 0.0d0
   do l = 1, nc
      t7 = t7 + n(l)*t4(l)
   end do

   t8 = 1d0/t5

   t9 = V - 0.414213562373095d0*t7*t8

   t10 = V + 2.41421356237309d0*t7*t8

   t11 = 1d0/t9

   t12 = 1d0/V

   t13 = 1d0/t7

   t14 = -t12*t7*t8 + 1.0d0

   t15 = log(t10*t11)

   t16 = t9**(-2)

   t17 = t5**(-2)

   t18 = log(t14)

   t19 = 1d0/t14

   t20 = 1d0/t10

   t23 = 2*R*T*t12*t19 - 0.323314343862425d0*t13*t20*t5*t6*t9*( &
      0.414213562373095d0*t10*t16*t8 + 2.41421356237309d0*t11*t8) + &
      0.323314343862425d0*t15*t5*t6/t7**2

   t24 = t13*t15*t5

   t25 = -R*T*t12*t19*t7*t8 - R*T*t18 - 0.161657171931213d0*t13*t15*t6 - &
      0.161657171931213d0*t13*t20*t5*t6*t9*(-0.414213562373095d0*t10* &
      t16*t17*t7 - 2.41421356237309d0*t11*t17*t7)

   res = -R*T*t18*t5 - 0.161657171931213d0*t13*t15*t5*t6
   f = res

   do i = 1, nc
      res = t23*t4(i) - 0.323314343862425d0*t24*t3(i) + t25
      dfdn(i) = res
   end do

end subroutine pr76_c6n_w3
