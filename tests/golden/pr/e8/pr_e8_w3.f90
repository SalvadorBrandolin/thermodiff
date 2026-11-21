subroutine pr_e8_w3(nc, V, T, Tc, ac, b, k, n, f, dfdn)
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
   real(8) :: res, t10, t11, t12, t13, t14, t4, t5, t7, t9
   real(8) :: t1(nc), t2(nc), t3(nc), t6(nc), t8(nc)
   integer :: i, l, m, p

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      do m = 1, nc
         t6(m) = 0.0d0
         do l = 1, nc
            t6(m) = t6(m) + (1 - k(l, m))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      t7 = 0.0d0
      do l = 1, nc
         t7 = t7 + t1(l)
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         t8(i) = 0.0d0
         do m = 1, nc
            t8(i) = t8(i) + (1 - k(i, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      t9 = 0.0d0
      do m = 1, nc
         t9 = t9 + n(m)*t2(m)*t3(m)*t6(m)
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      t10 = V - t7
   end if

   if (present(f) .or. present(dfdn)) then
      t11 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdn)) then
      t12 = V*(V + t7) + t10*t7
   end if

   if (present(f) .or. present(dfdn)) then
      t13 = 1d0/t12
   end if

   if (present(dfdn)) then
      t14 = R*T/t10**2 - t11*t9*(-2*V + 2*t7)/t12**2
   end if

   if (present(f)) then
      res = R*T/t10 - t11*t13*t9
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t13*(t11*t2(i)*t3(i)*t6(i) + t11*t2(i)*t3(i)*t8(i)) + t14*b(i)
         dfdn(i) = res
      end do
   end if

end subroutine pr_e8_w3
