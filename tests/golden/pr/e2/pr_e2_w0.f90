subroutine pr_e2_w0(nc, V, T, Tc, ac, b, k, n, f)
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
   real(8) :: res, t3, t4, t5
   real(8) :: t1(nc), t2(nc)
   integer :: l, m, p

   if (present(f)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f)) then
      t3 = 0.0d0
      do l = 1, nc
         t3 = t3 + t1(l)
      end do
   end if

   if (present(f)) then
      t4 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t4 = t4 + sqrt(T**4*ac(l)*ac(m)*t2(l)*t2(m))*(1 - k(l, m))*n(l)*n(m)
         end do
      end do
   end if

   if (present(f)) then
      t5 = V - t3
   end if

   if (present(f)) then
      res = R*T/t5 - t4/(V*(V + t3) + t3*t5)
      f = res
   end if

end subroutine pr_e2_w0
