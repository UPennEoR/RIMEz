module dfitpack_wrappers
    ! import c binding
    use ISO_C_BINDING, only: c_double


    implicit none


contains


    subroutine bispeu_wrap(tx,nx,ty,ny,c,kx,ky,x,y,z,m,wrk,lwrk,ier) bind(c)
      integer(4), intent(in) :: nx,ny,kx,ky,m,lwrk
      real(8),    intent(in) :: tx(nx),ty(ny),c((nx-kx-1)*(ny-ky-1)),&
          x(m),y(m),wrk(lwrk)
      integer(4), intent(out) :: ier
      real(8),    intent(out) :: z(m)
      call bispeu(tx,nx,ty,ny,c,kx,ky,x,y,z,m,wrk,lwrk,ier)
    end subroutine bispeu_wrap


end module dfitpack_wrappers
