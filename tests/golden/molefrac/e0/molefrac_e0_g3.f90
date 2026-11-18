subroutine molefrac_e0_g3(nc, n, f, dfdn, dfdndn)
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: n(:)
   real(8), optional, intent(out) :: f(:)
   real(8), optional, intent(out) :: dfdn(:, :)
   real(8), optional, intent(out) :: dfdndn(:, :, :)
   real(8) :: res, t1, t2, t3
   integer :: i, j, k, l

   if (present(f)) then
      do l = 1, nc
         t1 = 0.0d0
         do k = 1, nc
            t1 = t1 + n(k)
         end do
         res = n(l)/t1
         f(l) = res
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         do l = 1, nc
            t2 = 0.0d0
            do k = 1, nc
               t2 = t2 + n(k)
            end do
            if (i == l) then
               res = 1d0/t2 - n(i)/t2**2
            else
               res = -n(l)/t2**2
            end if
            dfdn(i, l) = res
         end do
      end do
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = 1, nc
            do l = 1, nc
               t3 = 0.0d0
               do k = 1, nc
                  t3 = t3 + n(k)
               end do
               if (i == j .and. i == l) then
                  res = -2/t3**2 + 2*n(i)/t3**3
               else if (i == l) then
                  res = -1/t3**2 + 2*n(i)/t3**3
               else if (j == l) then
                  res = -1/t3**2 + 2*n(j)/t3**3
               else
                  res = 2*n(l)/t3**3
               end if
               dfdndn(i, j, l) = res
            end do
         end do
      end do
   end if

end subroutine molefrac_e0_g3
