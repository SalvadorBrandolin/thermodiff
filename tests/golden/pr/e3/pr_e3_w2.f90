subroutine pr_e3_w2(nc, V, T, Tc, ac, b, k, n, f, dfdT)
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
   real(8), optional, intent(out) :: dfdT
   real(8) :: res, t4, t5, t6, t7, t8, t9
   real(8) :: t1(nc), t2(nc)
   real(8) :: t3(nc, nc)
   integer :: l, m, p, q

   if (present(f) .or. present(dfdT)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))
         end do
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      t4 = 0.0d0
      do l = 1, nc
         t4 = t4 + t1(l)
      end do
   end if

   if (present(f)) then
      t5 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t5 = t5 + (1 - k(l, m))*n(l)*n(m)*t3(l, m)
         end do
      end do
   end if

   if (present(dfdT)) then
      t6 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t6 = t6 + 2*(1 - k(l, m))*n(l)*n(m)*t3(l, m)/T
         end do
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      t7 = V - t4
   end if

   if (present(f) .or. present(dfdT)) then
      t8 = 1d0/(V*(V + t4) + t4*t7)
   end if

   if (present(f) .or. present(dfdT)) then
      t9 = 1d0/t7
   end if

   if (present(f)) then
      res = R*T*t9 - t5*t8
      f = res
   end if

   if (present(dfdT)) then
      res = R*t9 - t6*t8
      dfdT = res
   end if

end subroutine pr_e3_w2
