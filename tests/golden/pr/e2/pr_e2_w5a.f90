subroutine pr_e2_w5a(nc, V, T, Tc, ac, b, k, n, f, dfdV, dfdV2)
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
   real(8), optional, intent(out) :: dfdV
   real(8), optional, intent(out) :: dfdV2
   real(8) :: res, t4, t5, t6, t7, t8, t9
   real(8) :: t1(nc), t2(nc)
   real(8) :: t3(nc, nc)
   integer :: l, m, p, q

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))*(1 - k(p, q))*n(p)*n(q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t4 = 0.0d0
      do l = 1, nc
         t4 = t4 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t5 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t5 = t5 + t3(l, m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t6 = V - t4
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t7 = V*(V + t4) + t4*t6
   end if

   if (present(dfdV) .or. present(dfdV2)) then
      t8 = -2*V - 2*t4
   end if

   if (present(dfdV) .or. present(dfdV2)) then
      t9 = t7**(-2)
   end if

   if (present(f)) then
      res = R*T/t6 - t5/t7
      f = res
   end if

   if (present(dfdV)) then
      res = -R*T/t6**2 - t5*t8*t9
      dfdV = res
   end if

   if (present(dfdV2)) then
      res = 2*R*T/t6**3 + 2*t5*t9 - t5*t8*(-4*V - 4*t4)/t7**3
      dfdV2 = res
   end if

end subroutine pr_e2_w5a
