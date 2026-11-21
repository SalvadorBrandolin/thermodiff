subroutine pr_e2_w3(nc, V, T, Tc, ac, b, k, n, f, dfdn)
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
   real(8) :: res, t10, t11, t12, t6, t7
   real(8) :: t1(nc), t2(nc), t8(nc), t9(nc)
   real(8) :: t3(nc, nc), t4(nc, nc), t5(nc, nc)
   integer :: i, l, m, p, q

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))
         end do
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         do q = 1, nc
            t4(p, q) = 1 - k(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      do p = 1, nc
         do q = 1, nc
            t5(p, q) = n(p)*n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      t6 = 0.0d0
      do l = 1, nc
         t6 = t6 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      t7 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t7 = t7 + t5(l, m)
         end do
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         t8(i) = 0.0d0
         do l = 1, nc
            t8(i) = t8(i) + (1 - k(l, i))*n(l)*t3(i, l)
         end do
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         t9(i) = 0.0d0
         do m = 1, nc
            t9(i) = t9(i) + n(m)*t3(i, m)*t4(i, m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn)) then
      t10 = V - t6
   end if

   if (present(f) .or. present(dfdn)) then
      t11 = V*(V + t6) + t10*t6
   end if

   if (present(f) .or. present(dfdn)) then
      t12 = 1d0/t11
   end if

   if (present(f)) then
      res = R*T/t10 - t12*t7
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = R*T*b(i)/t10**2 - t12*(t8(i) + t9(i)) - t7*(-V*b(i) - t10*b(i) + &
               t6*b(i))/t11**2
         dfdn(i) = res
      end do
   end if

end subroutine pr_e2_w3
