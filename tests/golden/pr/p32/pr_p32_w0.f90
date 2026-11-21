subroutine pr_p32_w0(nc, V, T, Tc, ac, b, k, n, f)
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
   real(8) :: res, t4, t5, t6
   real(8) :: t1(nc), t2(nc), t3(nc)
   integer :: l, m, p

   if (present(f)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f)) then
      t4 = 0.0d0
      do l = 1, nc
         t4 = t4 + t1(l)
      end do
   end if

   if (present(f)) then
      t5 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t5 = t5 + (1 - k(l, m))*n(l)*n(m)*t2(l)*t2(m)*t3(l)*t3(m)
         end do
      end do
   end if

   if (present(f)) then
      t6 = V - t4
   end if

   if (present(f)) then
      res = R*T/t6 - t5*sqrt(T**4)/(V*(V + t4) + t4*t6)
      f = res
   end if

end subroutine pr_p32_w0
