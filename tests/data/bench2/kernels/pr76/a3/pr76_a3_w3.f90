subroutine pr76_a3_w3(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdn)
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
   real(8) :: res, t11, t12, t15, t18, t19, t20, t21, t22, t23, t24, t25, t26
   real(8) :: t1(nc), t13(nc), t14(nc), t16(nc), t17(nc), t2(nc), t5(nc), t6(nc)
   real(8) :: t10(nc, nc), t3(nc, nc), t4(nc, nc), t7(nc, nc), t8(nc, nc), t9(nc, nc)
   integer :: i, k, l, m, p, q

   do p = 1, nc
      t1(p) = 1d0/Pc(p)
   end do

   do p = 1, nc
      t2(p) = 0.038898035d0*R*Tc(p)*t1(p)
   end do

   do p = 1, nc
      do q = 1, nc
         t3(p, q) = t2(p) + t2(q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t4(p, q) = 1.0d0 - lij(p, q)
      end do
   end do

   do p = 1, nc
      t5(p) = ((1 - sqrt(T/Tc(p)))*(-0.26993d0*w(p)**2 + 1.54226d0*w(p) + 0.37464d0) + &
      1)**2
   end do

   do p = 1, nc
      t6(p) = Tc(p)**2
   end do

   do p = 1, nc
      do q = 1, nc
         t7(p, q) = sqrt(R**4*t1(p)*t1(q)*t5(p)*t5(q)*t6(p)*t6(q))
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t8(p, q) = 1.0d0 - kij(p, q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t9(p, q) = 1.0d0 - lij(q, p)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t10(p, q) = 0.45723553d0*n(p)*n(q)*t7(p, q)*t8(p, q)
      end do
   end do

   t11 = 0.0d0
   do m = 1, nc
      t11 = t11 + n(m)
   end do

   t12 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t12 = t12 + t10(k, l)
      end do
   end do

   do i = 1, nc
      t13(i) = 0.0d0
      do k = 1, nc
         t13(i) = t13(i) + 0.45723553d0*(1.0d0 - kij(k, i))*n(k)*t7(i, k)
      end do
   end do

   do i = 1, nc
      t14(i) = 0.0d0
      do l = 1, nc
         t14(i) = t14(i) + 0.45723553d0*n(l)*t7(i, l)*t8(i, l)
      end do
   end do

   t15 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t15 = t15 + n(k)*n(l)*t3(k, l)*t4(k, l)/t11
      end do
   end do

   do i = 1, nc
      t16(i) = 0.0d0
      do k = 1, nc
         t16(i) = t16(i) + n(k)*t3(i, k)*t9(i, k)/t11
      end do
   end do

   do i = 1, nc
      t17(i) = 0.0d0
      do l = 1, nc
         t17(i) = t17(i) + n(l)*t3(i, l)*t4(i, l)/t11
      end do
   end do

   t18 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t18 = t18 + -n(k)*n(l)*t3(k, l)*t4(k, l)/t11**2
      end do
   end do

   t19 = V - 0.414213562373095d0*t15

   t20 = V + 2.41421356237309d0*t15

   t21 = 1d0/t19

   t22 = 1d0/V

   t23 = log(t20*t21)

   t24 = -t15*t22 + 1.0d0

   t25 = 1d0/t15

   t26 = log(t24)

   res = -R*T*t11*t26 - 0.353553390593274d0*t12*t23*t25
   f = res

   do i = 1, nc
      res = R*T*t11*t22*(t18 + t16(i) + t17(i))/t24 - R*T*t26 - &
            0.353553390593274d0*t12*t19*t25*(2.41421356237309d0*t21*(t18 + &
            t16(i) + t17(i)) + 0.414213562373095d0*t20*(t18 + t16(i) + t17(i &
            ))/t19**2)/t20 + 0.353553390593274d0*t12*t23*(t18 + t16(i) + t17( &
            i))/t15**2 - 0.353553390593274d0*t23*t25*(t13(i) + t14(i))
      dfdn(i) = res
   end do

end subroutine pr76_a3_w3
