subroutine pr_p32_w5d(nc, V, T, Tc, ac, b, k, n, f, dfdn, dfdndT, dfdndV)
   use yaeos__constants, only: R
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), contiguous, intent(in) :: Tc(:)
   real(8), contiguous, intent(in) :: ac(:)
   real(8), contiguous, intent(in) :: b(:)
   real(8), contiguous, intent(in) :: k(:, :)
   real(8), contiguous, intent(in) :: n(:)
   real(8), optional, intent(out) :: f
   real(8), contiguous, optional, intent(out) :: dfdn(:)
   real(8), contiguous, optional, intent(out) :: dfdndT(:)
   real(8), contiguous, optional, intent(out) :: dfdndV(:)
   real(8) :: res, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t4, t5, t7, t9
   real(8) :: t1(nc), t2(nc), t3(nc), t6(nc), t8(nc)
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
      do l = 1, nc
         t6(l) = 0.0d0
      end do
      do m = 1, nc
         do l = 1, nc
            t6(l) = t6(l) + (1 - k(l, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t7 = 0.0d0
      do l = 1, nc
         t7 = t7 + t1(l)
      end do
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do i = 1, nc
         t8(i) = 0.0d0
         do l = 1, nc
            t8(i) = t8(i) + (1 - k(l, i))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t9 = 0.0d0
      do l = 1, nc
         t9 = t9 + n(l)*t2(l)*t3(l)*t6(l)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t10 = V - t7
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t11 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t12 = V*(V + t7) + t10*t7
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t13 = t12**(-2)
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t14 = -2*V
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t15 = t14 + 2*t7
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT)) then
      t16 = 1d0/t12
   end if

   if (present(dfdndT)) then
      t17 = 1d0/T
   end if

   if (present(dfdn) .or. present(dfdndT)) then
      t18 = t10**(-2)
   end if

   if (present(dfdn)) then
      t19 = R*T*t18 - t11*t13*t15*t9
   end if

   if (present(dfdndT)) then
      t20 = R*t18 - 2*t11*t13*t15*t17*t9
   end if

   if (present(dfdndT)) then
      t21 = t11*t17
   end if

   if (present(dfdndV)) then
      t22 = -2*R*T/t10**3 + 2*t11*t13*t9 - t11*t15*t9*(-4*V - 4*t7)/t12**3
   end if

   if (present(dfdndV)) then
      t23 = t13*(t14 - 2*t7)
   end if

   if (present(f)) then
      res = R*T/t10 - t11*t16*t9
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t16*(t11*t2(i)*t3(i)*t6(i) + t11*t2(i)*t3(i)*t8(i)) + t19*b(i)
         dfdn(i) = res
      end do
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         res = -t16*(2*t21*t2(i)*t3(i)*t6(i) + 2*t21*t2(i)*t3(i)*t8(i)) + t20*b(i &
               )
         dfdndT(i) = res
      end do
   end if

   if (present(dfdndV)) then
      do i = 1, nc
         res = t22*b(i) - t23*(t11*t2(i)*t3(i)*t6(i) + t11*t2(i)*t3(i)*t8(i))
         dfdndV(i) = res
      end do
   end if

end subroutine pr_p32_w5d
