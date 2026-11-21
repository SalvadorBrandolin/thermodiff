subroutine pr_e0_w5c(nc, V, T, Tc, ac, b, k, n, f, dfdn, dfdndn)
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
   real(8), optional, intent(out) :: dfdndn(:, :)
   real(8) :: res, t1, t10, t11, t12, t2, t3, t4, t5, t6, t7, t8, t9
   integer :: i, j, l, m

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

   if (present(dfdn)) then
      do i = 1, nc
         t3 = 0.0d0
         do l = 1, nc
            t3 = t3 + b(l)*n(l)
         end do
         t4 = 0.0d0
         do l = 1, nc
            t4 = t4 + sqrt(T**4*ac(i)*ac(l)/(Tc(i)**2*Tc(l)**2))*(1 - k(l, i))*n(l)
         end do
         t5 = 0.0d0
         do m = 1, nc
            t5 = t5 + sqrt(T**4*ac(i)*ac(m)/(Tc(i)**2*Tc(m)**2))*(1 - k(i, m))*n(m)
         end do
         t6 = 0.0d0
         do m = 1, nc
            do l = 1, nc
               t6 = t6 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
            end do
         end do
         res = R*T*b(i)/(V - t3)**2 - t6*(-V*b(i) + t3*b(i) - (V - t3)*b(i))/(V*( &
               V + t3) + t3*(V - t3))**2 - (t4 + t5)/(V*(V + t3) + t3*(V - t3))
         dfdn(i) = res
      end do
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = 1, nc
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
            do l = 1, nc
               t10 = t10 + sqrt(T**4*ac(j)*ac(l)/(Tc(j)**2*Tc(l)**2))*(1 - k(l, j))*n(l)
            end do
            t11 = 0.0d0
            do m = 1, nc
               t11 = t11 + sqrt(T**4*ac(j)*ac(m)/(Tc(j)**2*Tc(m)**2))*(1 - k(j, m))*n(m)
            end do
            t12 = 0.0d0
            do m = 1, nc
               do l = 1, nc
                  t12 = t12 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
               end do
            end do
            res = 2*R*T*b(i)*b(j)/(V - t7)**3 - 2*t12*b(i)*b(j)/(V*(V + t7) + t7*(V &
                  - t7))**2 - t12*(-V*b(i) + t7*b(i) - (V - t7)*b(i))*(-2*V*b(j) + &
                  2*t7*b(j) - 2*(V - t7)*b(j))/(V*(V + t7) + t7*(V - t7))**3 - (t10 &
                  + t11)*(-V*b(i) + t7*b(i) - (V - t7)*b(i))/(V*(V + t7) + t7*(V - &
                  t7))**2 - (t8 + t9)*(-V*b(j) + t7*b(j) - (V - t7)*b(j))/(V*(V + &
                  t7) + t7*(V - t7))**2 - (sqrt(T**4*ac(i)*ac(j)/(Tc(i)**2*Tc(j)**2 &
                  ))*(1 - k(i, j)) + sqrt(T**4*ac(i)*ac(j)/(Tc(i)**2*Tc(j)**2))*(1 &
                  - k(j, i)))/(V*(V + t7) + t7*(V - t7))
            dfdndn(i, j) = res
         end do
      end do
   end if

end subroutine pr_e0_w5c
