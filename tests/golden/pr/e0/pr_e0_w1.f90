subroutine pr_e0_w1(nc, V, T, Tc, ac, b, k, n, f, dfdV)
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
   real(8) :: res, t1, t2, t3, t4
   integer :: l, m

   if (present(f)) then
      t1 = 0.0d0
      do l = 1, nc
         t1 = t1 + b(l)*n(l)
      end do
      t2 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t2 = t2 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
         end do
      end do
      res = R*T/(V - t1) - t2/(V*(V + t1) + t1*(V - t1))
      f = res
   end if

   if (present(dfdV)) then
      t3 = 0.0d0
      do l = 1, nc
         t3 = t3 + b(l)*n(l)
      end do
      t4 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t4 = t4 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
         end do
      end do
      res = -R*T/(V - t3)**2 - t4*(-2*V - 2*t3)/(V*(V + t3) + t3*(V - t3))**2
      dfdV = res
   end if

end subroutine pr_e0_w1
