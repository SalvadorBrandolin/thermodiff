subroutine pr_e0_w4(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdn)
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
   real(8), optional, intent(out) :: dfdV
   real(8), optional, intent(out) :: dfdn(:)
   real(8) :: res, t1, t10, t2, t3, t4, t5, t6, t7, t8, t9
   integer :: i, l, m

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

   if (present(dfdT)) then
      t3 = 0.0d0
      do l = 1, nc
         t3 = t3 + b(l)*n(l)
      end do
      t4 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t4 = t4 + 2*sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)/T
         end do
      end do
      res = R/(V - t3) - t4/(V*(V + t3) + t3*(V - t3))
      dfdT = res
   end if

   if (present(dfdV)) then
      t5 = 0.0d0
      do l = 1, nc
         t5 = t5 + b(l)*n(l)
      end do
      t6 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t6 = t6 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
         end do
      end do
      res = -R*T/(V - t5)**2 - t6*(-2*V - 2*t5)/(V*(V + t5) + t5*(V - t5))**2
      dfdV = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         t7 = 0.0d0
         do l = 1, nc
            t7 = t7 + b(l)*n(l)
         end do
         t8 = 0.0d0
         do l = 1, nc
            t8 = t8 + sqrt(T**4*ac(i)*ac(l)/(Tc(i)**2*Tc(l)**2))*(1 - k(l, i))*n(l)
         end do
         t9 = 0.0d0
         do m = 1, nc
            t9 = t9 + sqrt(T**4*ac(i)*ac(m)/(Tc(i)**2*Tc(m)**2))*(1 - k(i, m))*n(m)
         end do
         t10 = 0.0d0
         do m = 1, nc
            do l = 1, nc
               t10 = t10 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
            end do
         end do
         res = R*T*b(i)/(V - t7)**2 - t10*(-V*b(i) + t7*b(i) - (V - t7)*b(i))/(V* &
               (V + t7) + t7*(V - t7))**2 - (t8 + t9)/(V*(V + t7) + t7*(V - t7))
         dfdn(i) = res
      end do
   end if

end subroutine pr_e0_w4
