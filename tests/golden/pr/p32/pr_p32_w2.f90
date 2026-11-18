subroutine pr_p32_w2(nc, V, T, Tc, ac, b, k, n, f, dfdT)
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
   real(8), optional, intent(out) :: dfdT
   real(8) :: res, t10, t11, t4, t6, t7, t8, t9
   real(8) :: t1(nc), t2(nc), t3(nc), t5(nc)
   integer :: l, m, p

   if (present(f) .or. present(dfdT)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      do l = 1, nc
         t5(l) = 0.0d0
      end do
      do m = 1, nc
         do l = 1, nc
            t5(l) = t5(l) + (1 - k(l, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      t6 = 0.0d0
      do l = 1, nc
         t6 = t6 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      t7 = 0.0d0
      do l = 1, nc
         t7 = t7 + n(l)*t2(l)*t3(l)*t5(l)
      end do
   end if

   if (present(f) .or. present(dfdT)) then
      t8 = V - t6
   end if

   if (present(f) .or. present(dfdT)) then
      t9 = 1d0/(V*(V + t6) + t6*t8)
   end if

   if (present(f) .or. present(dfdT)) then
      t10 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdT)) then
      t11 = 1d0/t8
   end if

   if (present(f)) then
      res = R*T*t11 - t10*t7*t9
      f = res
   end if

   if (present(dfdT)) then
      res = R*t11 - 2*t10*t7*t9/T
      dfdT = res
   end if

end subroutine pr_p32_w2
