subroutine molefrac_e0_g0(nc, n, f)
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: n(:)
   real(8), optional, intent(out) :: f(:)
   real(8) :: res, t1
   integer :: k, l

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

end subroutine molefrac_e0_g0
