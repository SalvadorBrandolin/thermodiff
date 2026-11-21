subroutine pr_e0_w6(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, dfdndV)
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
   real(8), optional, intent(out) :: dfdT2
   real(8), optional, intent(out) :: dfdV2
   real(8), optional, intent(out) :: dfdndn(:, :)
   real(8), optional, intent(out) :: dfdTdV
   real(8), optional, intent(out) :: dfdndT(:)
   real(8), optional, intent(out) :: dfdndV(:)
   real(8) :: res, t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t2, t20, t21, t22, t23, t24, t25, t26, t27, t28
   real(8) :: t29, t3, t30, t4, t5, t6, t7, t8, t9
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

   if (present(dfdT2)) then
      t11 = 0.0d0
      do l = 1, nc
         t11 = t11 + b(l)*n(l)
      end do
      t12 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t12 = t12 + 2*sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)/T** &
      2
         end do
      end do
      res = -t12/(V*(V + t11) + t11*(V - t11))
      dfdT2 = res
   end if

   if (present(dfdV2)) then
      t13 = 0.0d0
      do l = 1, nc
         t13 = t13 + b(l)*n(l)
      end do
      t14 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t14 = t14 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
         end do
      end do
      res = 2*R*T/(V - t13)**3 - t14*(-4*V - 4*t13)*(-2*V - 2*t13)/(V*(V + t13 &
            ) + t13*(V - t13))**3 + 2*t14/(V*(V + t13) + t13*(V - t13))**2
      dfdV2 = res
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = 1, nc
            t15 = 0.0d0
            do l = 1, nc
               t15 = t15 + b(l)*n(l)
            end do
            t16 = 0.0d0
            do l = 1, nc
               t16 = t16 + sqrt(T**4*ac(i)*ac(l)/(Tc(i)**2*Tc(l)**2))*(1 - k(l, i))*n(l)
            end do
            t17 = 0.0d0
            do m = 1, nc
               t17 = t17 + sqrt(T**4*ac(i)*ac(m)/(Tc(i)**2*Tc(m)**2))*(1 - k(i, m))*n(m)
            end do
            t18 = 0.0d0
            do l = 1, nc
               t18 = t18 + sqrt(T**4*ac(j)*ac(l)/(Tc(j)**2*Tc(l)**2))*(1 - k(l, j))*n(l)
            end do
            t19 = 0.0d0
            do m = 1, nc
               t19 = t19 + sqrt(T**4*ac(j)*ac(m)/(Tc(j)**2*Tc(m)**2))*(1 - k(j, m))*n(m)
            end do
            t20 = 0.0d0
            do m = 1, nc
               do l = 1, nc
                  t20 = t20 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
               end do
            end do
            res = 2*R*T*b(i)*b(j)/(V - t15)**3 - 2*t20*b(i)*b(j)/(V*(V + t15) + t15* &
                  (V - t15))**2 - t20*(-V*b(i) + t15*b(i) - (V - t15)*b(i))*(-2*V*b &
                  (j) + 2*t15*b(j) - 2*(V - t15)*b(j))/(V*(V + t15) + t15*(V - t15 &
                  ))**3 - (t16 + t17)*(-V*b(j) + t15*b(j) - (V - t15)*b(j))/(V*(V + &
                  t15) + t15*(V - t15))**2 - (t18 + t19)*(-V*b(i) + t15*b(i) - (V - &
                  t15)*b(i))/(V*(V + t15) + t15*(V - t15))**2 - (sqrt(T**4*ac(i)*ac &
                  (j)/(Tc(i)**2*Tc(j)**2))*(1 - k(i, j)) + sqrt(T**4*ac(i)*ac(j)/( &
                  Tc(i)**2*Tc(j)**2))*(1 - k(j, i)))/(V*(V + t15) + t15*(V - t15))
            dfdndn(i, j) = res
         end do
      end do
   end if

   if (present(dfdTdV)) then
      t21 = 0.0d0
      do l = 1, nc
         t21 = t21 + b(l)*n(l)
      end do
      t22 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t22 = t22 + 2*sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)/T
         end do
      end do
      res = -R/(V - t21)**2 - t22*(-2*V - 2*t21)/(V*(V + t21) + t21*(V - t21)) &
            **2
      dfdTdV = res
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         t23 = 0.0d0
         do l = 1, nc
            t23 = t23 + b(l)*n(l)
         end do
         t24 = 0.0d0
         do l = 1, nc
            t24 = t24 + 2*sqrt(T**4*ac(i)*ac(l)/(Tc(i)**2*Tc(l)**2))*(1 - k(l, i))*n(l)/T
         end do
         t25 = 0.0d0
         do m = 1, nc
            t25 = t25 + 2*sqrt(T**4*ac(i)*ac(m)/(Tc(i)**2*Tc(m)**2))*(1 - k(i, m))*n(m)/T
         end do
         t26 = 0.0d0
         do m = 1, nc
            do l = 1, nc
               t26 = t26 + 2*sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)/T
            end do
         end do
         res = R*b(i)/(V - t23)**2 - t26*(-V*b(i) + t23*b(i) - (V - t23)*b(i))/(V &
               *(V + t23) + t23*(V - t23))**2 - (t24 + t25)/(V*(V + t23) + t23*( &
               V - t23))
         dfdndT(i) = res
      end do
   end if

   if (present(dfdndV)) then
      do i = 1, nc
         t27 = 0.0d0
         do l = 1, nc
            t27 = t27 + b(l)*n(l)
         end do
         t28 = 0.0d0
         do l = 1, nc
            t28 = t28 + sqrt(T**4*ac(i)*ac(l)/(Tc(i)**2*Tc(l)**2))*(1 - k(l, i))*n(l)
         end do
         t29 = 0.0d0
         do m = 1, nc
            t29 = t29 + sqrt(T**4*ac(i)*ac(m)/(Tc(i)**2*Tc(m)**2))*(1 - k(i, m))*n(m)
         end do
         t30 = 0.0d0
         do m = 1, nc
            do l = 1, nc
               t30 = t30 + sqrt(T**4*ac(l)*ac(m)/(Tc(l)**2*Tc(m)**2))*(1 - k(l, m))*n(l)*n(m)
            end do
         end do
         res = -2*R*T*b(i)/(V - t27)**3 - t30*(-2*V - 2*t27)*(-2*V*b(i) + 2*t27*b &
               (i) - 2*(V - t27)*b(i))/(V*(V + t27) + t27*(V - t27))**3 + 2*t30* &
               b(i)/(V*(V + t27) + t27*(V - t27))**2 - (-2*V - 2*t27)*(t28 + t29 &
               )/(V*(V + t27) + t27*(V - t27))**2
         dfdndV(i) = res
      end do
   end if

end subroutine pr_e0_w6
