subroutine molefrac_e0_g1(nc, n, f, dfdn)
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: n(:)
   real(8), optional, intent(out) :: f(:)
   real(8), optional, intent(out) :: dfdn(:, :)
   real(8) :: res, t1, t2
   integer :: i, k, l

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

end subroutine molefrac_e0_g1
