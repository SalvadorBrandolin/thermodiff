subroutine pr_e8_w5d(nc, V, T, Tc, ac, b, k, n, f, dfdn, dfdndT, dfdndV)
   use yaeos__constants, only: R
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), intent(in) :: Tc(:)
   real(8), intent(in) :: ac(:)
   real(8), intent(in) :: b(:)
   real(8), intent(in) :: k(:, :)
   real(8), intent(in) :: n(:)
   real(8), optional, intent(out) :: f
   real(8), optional, intent(out) :: dfdn(:)
   real(8), optional, intent(out) :: dfdndT(:)
   real(8), optional, intent(out) :: dfdndV(:)
   real(8) :: res, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t4, t5, t6, t7, t9
   real(8) :: t1(nc), t10(nc), t2(nc), t3(nc), t8(nc)
   integer :: i, l, m, p

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do m = 1, nc
         t8(m) = 0.0d0
         do l = 1, nc
            t8(m) = t8(m) + (1 - k(l, m))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t9 = 0.0d0
      do l = 1, nc
         t9 = t9 + t1(l)
      end do
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do i = 1, nc
         t10(i) = 0.0d0
         do m = 1, nc
            t10(i) = t10(i) + (1 - k(i, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t11 = 0.0d0
      do m = 1, nc
         t11 = t11 + n(m)*t2(m)*t3(m)*t8(m)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t12 = V - t9
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t13 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t14 = V*(V + t9) + t12*t9
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t15 = t14**(-2)
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t16 = -2*V
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t17 = t16 + 2*t9
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT)) then
      t18 = 1d0/t14
   end if

   if (present(dfdndT)) then
      t19 = 1d0/T
   end if

   if (present(dfdn) .or. present(dfdndT)) then
      t20 = t12**(-2)
   end if

   if (present(dfdn)) then
      t21 = R*T*t20 - t11*t13*t15*t17
   end if

   if (present(dfdndT)) then
      t22 = R*t20 - 2*t11*t13*t15*t17*t19
   end if

   if (present(dfdndT)) then
      t23 = t13*t19
   end if

   if (present(dfdndV)) then
      t24 = -2*R*T/t12**3 + 2*t11*t13*t15 - t11*t13*t17*(-4*V - 4*t9)/t14**3
   end if

   if (present(dfdndV)) then
      t25 = t15*(t16 - 2*t9)
   end if

   if (present(f)) then
      res = R*T/t12 - t11*t13*t18
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t18*(t13*t10(i)*t2(i)*t3(i) + t13*t2(i)*t3(i)*t8(i)) + t21*b(i)
         dfdn(i) = res
      end do
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         res = -t18*(2*t23*t10(i)*t2(i)*t3(i) + 2*t23*t2(i)*t3(i)*t8(i)) + t22*b( &
               i)
         dfdndT(i) = res
      end do
   end if

   if (present(dfdndV)) then
      do i = 1, nc
         res = t24*b(i) - t25*(t13*t10(i)*t2(i)*t3(i) + t13*t2(i)*t3(i)*t8(i))
         dfdndV(i) = res
      end do
   end if

end subroutine pr_e8_w5d
