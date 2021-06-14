#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <fftw3.h>

#ifdef MPI_PARALLEL
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

using namespace std;

template<class T> inline T sqr(T x) {return x*x;}


void smoothing(int ng, double tin, int fsize, double *in, double *out);
void find_peaks_minima(double *kappa, double *gamma, int ng, int offset, double *peaks, double *minima,
                       double numin, double numax, int Nnu);
void calc_PDF(double *kappa, double *gamma, int ng, int offset, double *PDF, double numin, double numax, int Nnu);
double calc_kappa2(double *kappa, double *gamma, int ng);


int main(int argc, char *argv[]){
  FILE *fp;
  char rootdir[256], outroot[256], outdir[256], basedir[256], inname[256], outname[256];
  int myrank, numprocs;
  int Nnu, ng, dummy, fsize, offset;
  double pi, pix, tin, noise, numin, numax, dnu;
  double *kappa, *kappa_sm, *gamma, *gamma_sm, *data, *nu, *peaks, *minima, *PDF;
  double *peaks_m, *peaks_s, *minima_m, *minima_s, *PDF_m, *PDF_s;
  double *peaks_m0, *peaks_s0, *minima_m0, *minima_s0, *PDF_m0, *PDF_s0;
  double *kappa2, *kappa2_m, *kappa2_s, *kappa2_m0, *kappa2_s0;


  if(argc!=6){
    printf("usage:./%s [rootdir] [outroot] [outdir] [smoothing] [nu binning]\n", argv[0]);
    exit(1);
  }

  const double n_gal = 30.0; // number density of galaxies [arcmin^-2]
  const double sigma_g = 0.4; // variance of intrinsic shear
  const double theta = 5.0; // opening angle [deg]
  const int Nz = 40;
  const int NR = 10000;
  //const int NR = 50;


#ifdef MPI_PARALLEL
  mpi::environment env(argc, argv);
  mpi::communicator world;

  myrank = world.rank();
  numprocs = world.size();

  if(myrank == 0) cout << "[NOTE] MPI parallel mode" << endl;
  world.barrier();
  cout << "myrank/numprocs: " << myrank << "/" << numprocs << endl;
#else
  myrank = 0;
  numprocs = 1;
  cout << "[NOTE] serial mode" << endl;
#endif


  sprintf(rootdir, "%s", argv[1]);
  sprintf(outroot, "%s", argv[2]);
  sprintf(outdir, "%s", argv[3]);

  const double theta_G = atof(argv[4]); // smoothing scale [arcmin]
  const int nubin = atoi(argv[5]);

  if(nubin == 0){
    /* For PDF */
    numin = -0.25;
    numax = 1.00;
    Nnu = 500;
  }

  if(nubin == 1){
    /* For peaks */
    numin = -0.25;
    numax = 0.50;
    Nnu = 300;
  }

  if(nubin == 2){
    /* For minima */
    numin = -0.25;
    numax = 0.25;
    Nnu = 200;
  }


  pi = 4.0*atan(1.0);


  if(myrank == 0){
    peaks_m = new double[Nnu];
    minima_m = new double[Nnu];
    PDF_m = new double[Nnu];
    kappa2_m = new double;
    peaks_s = new double[Nnu];
    minima_s = new double[Nnu];
    PDF_s = new double[Nnu];
    kappa2_s = new double;
  }

  peaks_m0 = new double[Nnu];
  minima_m0 = new double[Nnu];
  PDF_m0 = new double[Nnu];
  kappa2_m0 = new double;
  peaks_s0 = new double[Nnu];
  minima_s0 = new double[Nnu];
  PDF_s0 = new double[Nnu];
  kappa2_s0 = new double;

  nu = new double[Nnu];
  dnu = (numax-numin)/((double) Nnu);
  for(int i=0;i<Nnu;++i){
    nu[i] = dnu*(i+0.5) + numin;
  }


  for(int iz=1;iz<=Nz;++iz){
    if(myrank == 0){
      for(int i=0;i<Nnu;++i){
        peaks_m[i] = 0.0;
        minima_m[i] = 0.0;
        PDF_m[i] = 0.0;
        peaks_s[i] = 0.0;
        minima_s[i] = 0.0;
        PDF_s[i] = 0.0;
      }
      *kappa2_m = 0.0;
      *kappa2_s = 0.0;
    }

    for(int i=0;i<Nnu;++i){
      peaks_m0[i] = 0.0;
      minima_m0[i] = 0.0;
      PDF_m0[i] = 0.0;
      peaks_s0[i] = 0.0;
      minima_s0[i] = 0.0;
      PDF_s0[i] = 0.0;
    }
    *kappa2_m0 = 0.0;
    *kappa2_s0 = 0.0;


    for(int iR=0;iR<NR;++iR){
      if(iR % numprocs != myrank) continue;

      sprintf(basedir, "%s/LP%03d/run%03d", rootdir, iR/100+1, iR%100+1);
      //sprintf(basedir, "%s/run%03d", rootdir, iR+1);

      // read data for kappa
      sprintf(inname, "%s/kappa%02d.dat", basedir, iz);
      cout << "input file name:" << inname << endl;

      if((fp = fopen(inname, "rb"))==NULL){
        printf("can't open the files!\n");
        exit(1);
      }

      fread(&dummy, sizeof(int), 1, fp);
      ng = dummy;
      kappa = new double[ng*ng];
      kappa_sm = new double[ng*ng];

      fread(kappa, sizeof(double), ng*ng, fp);
      fread(&dummy, sizeof(int), 1, fp);

      if(ng != dummy){
        printf("loading failed!\n");
        exit(1);
      }

      fclose(fp);

      // read data for gamma1/2
      sprintf(inname, "%s/gamma%02d.dat", basedir, iz);
      cout << "input file name:" << inname << endl;

      if((fp = fopen(inname, "rb"))==NULL){
        printf("can't open the files!\n");
        exit(1);
      }

      fread(&dummy, sizeof(int), 1, fp);
      ng = dummy;
      data = new double[2*ng*ng];
      gamma = new double[ng*ng];
      gamma_sm = new double[ng*ng];

      fread(data, sizeof(double), 2*ng*ng, fp);
      fread(&dummy, sizeof(int), 1, fp);

      if(ng != dummy){
        printf("loading failed!:%d\n", dummy);
        exit(1);
      }

      fclose(fp);

      for(int i=0;i<ng*ng;++i){
        gamma[i] = sqrt(sqr(data[0+2*i]) + sqr(data[1+2*i]));
      }

      delete[] data;


      pix = theta/ng; // pixel size
      tin = (theta_G/60.0)/pix; // smoothing scale in pixel unit
      noise = sqrt(sqr(sigma_g)/2.0/(2.0*pi*n_gal*sqr(theta_G))); // noise due to intrinsic elipcity
      fsize = (int) ceil(10.0*tin); // smoothing kernel range is 10-sigma
      fsize = 2*(fsize/2)+1; // fsize should be odd
      offset = 2*((int) ceil(tin)); // 2*theta_G cannot be used due to incomplete smoothing

      if(myrank == 0){
        printf("pixel:%g [arcmin], offset:%g [arcmin] (%d pixels)\n", pix*60.0, offset*pix*60.0, offset);
        printf("effective area:%e [deg^2]\n", sqr(theta-2*offset*pix));
      }

      smoothing(ng, tin, fsize, kappa, kappa_sm);
      smoothing(ng, tin, fsize, gamma, gamma_sm);


      peaks = new double[Nnu];
      minima = new double[Nnu];
      find_peaks_minima(kappa_sm, gamma_sm, ng, offset, peaks, minima, numin, numax, Nnu);

      PDF = new double[Nnu];
      calc_PDF(kappa_sm, gamma_sm, ng, offset, PDF, numin, numax, Nnu);
      //calc_PDF(kappa, gamma, ng, PDF, numin, numax, Nnu);

      kappa2 = new double;
      *kappa2 = calc_kappa2(kappa, gamma, ng);


      sprintf(outname, "%s/%s/pm%02d.dat", basedir, outdir, iz);

      if((fp = fopen(outname, "wb")) == NULL){
        printf("file open error!:%s\n", outname);
        exit(1);
      }


      fwrite(&Nnu, sizeof(int), 1, fp);
      fwrite(nu, sizeof(double), Nnu, fp);
      fwrite(&Nnu, sizeof(int), 1, fp);

      fwrite(&Nnu, sizeof(int), 1, fp);
      fwrite(peaks, sizeof(double), Nnu, fp);
      fwrite(&Nnu, sizeof(int), 1, fp);

      fwrite(&Nnu, sizeof(int), 1, fp);
      fwrite(minima, sizeof(double), Nnu, fp);
      fwrite(&Nnu, sizeof(int), 1, fp);

      fwrite(&Nnu, sizeof(int), 1, fp);
      fwrite(PDF, sizeof(double), Nnu, fp);
      fwrite(&Nnu, sizeof(int), 1, fp);

      fwrite(&Nnu, sizeof(int), 1, fp);
      fwrite(kappa2, sizeof(double), 1, fp);
      fwrite(&Nnu, sizeof(int), 1, fp);

      fclose(fp);


      for(int i=0;i<Nnu;++i){
        peaks_m0[i] += peaks[i];
        minima_m0[i] += minima[i];
        PDF_m0[i] += PDF[i];
        peaks_s0[i] += sqr(peaks[i]);
        minima_s0[i] += sqr(minima[i]);
        PDF_s0[i] += sqr(PDF[i]);
      }
      *kappa2_m0 += *kappa2;
      *kappa2_s0 += sqr(*kappa2);


      delete[] kappa;
      delete[] kappa_sm;
      delete[] gamma;
      delete[] gamma_sm;
      delete[] peaks;
      delete[] minima;
      delete[] PDF;
      delete kappa2;
    }


#ifdef MPI_PARALLEL
    if(myrank == 0){
      reduce(world, peaks_m0, Nnu, peaks_m, plus<double>(), 0);
      reduce(world, minima_m0, Nnu, minima_m, plus<double>(), 0);
      reduce(world, PDF_m0, Nnu, PDF_m, plus<double>(), 0);
      reduce(world, kappa2_m0, 1, kappa2_m, plus<double>(), 0);
    }
    else{
      reduce(world, peaks_m0, Nnu, plus<double>(), 0);
      reduce(world, minima_m0, Nnu, plus<double>(), 0);
      reduce(world, PDF_m0, Nnu, plus<double>(), 0);
      reduce(world, kappa2_m0, 1, plus<double>(), 0);
    }

    if(myrank == 0){
      reduce(world, peaks_s0, Nnu, peaks_s, plus<double>(), 0);
      reduce(world, minima_s0, Nnu, minima_s, plus<double>(), 0);
      reduce(world, PDF_s0, Nnu, PDF_s, plus<double>(), 0);
      reduce(world, kappa2_s0, 1, kappa2_s, plus<double>(), 0);
    }
    else{
      reduce(world, peaks_s0, Nnu, plus<double>(), 0);
      reduce(world, minima_s0, Nnu, plus<double>(), 0);
      reduce(world, PDF_s0, Nnu, plus<double>(), 0);
      reduce(world, kappa2_s0, 1, plus<double>(), 0);
    }
#endif

    if(myrank == 0){
      cout << iz << "-th peaks/minima counts done. Compute mean and std. dev." << endl;

      for(int i=0;i<Nnu;++i){
        peaks_m[i] /= (double) NR;
        minima_m[i] /= (double) NR;
        PDF_m[i] /= (double) NR;

        peaks_s[i] /= (double) NR;
        minima_s[i] /= (double) NR;
        PDF_s[i] /= (double) NR;

        peaks_s[i] = sqrt((((double) NR)/(NR-1.0))*(peaks_s[i]-sqr(peaks_m[i])));
        minima_s[i] = sqrt((((double) NR)/(NR-1.0))*(minima_s[i]-sqr(minima_m[i])));
        PDF_s[i] = sqrt((((double) NR)/(NR-1.0))*(PDF_s[i]-sqr(PDF_m[i])));
      }

      *kappa2_m /= (double) NR;
      *kappa2_s /= (double) NR;
      *kappa2_s = sqrt((((double) NR)/(NR-1.0))*(*kappa2_s-sqr(*kappa2_m)));


      sprintf(outname, "%s/%s/pm%02d.dat", outroot, outdir, iz);

      if((fp = fopen(outname, "w")) == NULL){
        printf("file open error!:%s\n", outname);
        exit(1);
      }

      fprintf(fp, "#nu peaks std minima std PDF std kappa2 std\n");
      fprintf(fp, "#effective area:%e [deg^2]\n", sqr(theta-2.0*offset*pix));

      for(int i=0;i<Nnu;++i){
        fprintf(fp, "%e %e %e %e %e %e %e %e %e\n", nu[i], peaks_m[i], peaks_s[i],
                                                    minima_m[i], minima_s[i],
                                                    PDF_m[i], PDF_s[i],
                                                    *kappa2_m, *kappa2_s);
      }

      fclose(fp);
    }

#ifdef MPI_PARALLEL
    world.barrier();
#endif
  }

  if(myrank == 0){
    delete[] peaks_m;
    delete[] peaks_s;
    delete[] minima_m;
    delete[] minima_s;
    delete[] PDF_m;
    delete[] PDF_s;
    delete kappa2_m;
    delete kappa2_s;
  }

  delete[] peaks_m0;
  delete[] peaks_s0;
  delete[] minima_m0;
  delete[] minima_s0;
  delete[] PDF_m0;
  delete[] PDF_s0;
  delete kappa2_m0;
  delete kappa2_s0;
  delete[] nu;

  if(myrank == 0) cout << "Success" << endl;


  return 0;
}

void smoothing(int ng, double tin, int fsize, double *in, double *out){
  int i, j, k, nf, np;
  double xx, yy, rr, tt, f1, f2, re, im;
  double *fil1;
  fftw_complex *fft_gam, *fft_fil;
  fftw_plan fft_p;


  nf = fsize/2;
  np = ng+nf;

  fft_gam = (fftw_complex *) fftw_malloc(np*np*sizeof(fftw_complex));
  fft_fil = (fftw_complex *) fftw_malloc(np*np*sizeof(fftw_complex));

  if((fft_gam == NULL) || (fft_fil == NULL)){
    fprintf(stderr, "memory allocation failed\n");
    exit(1);
  }

  // FFT filter
  fil1 = new double[fsize*fsize];


  f2 = 0.0;
  for(j=0;j<fsize;j++){
    for(i=0;i<fsize;i++){
      xx = ((double)(i-nf));
      yy = ((double)(j-nf));
      rr = sqrt(xx*xx+yy*yy);
      f1 = exp(-rr*rr/tin/tin);
      fil1[i+j*fsize]= f1;
      f2 = f2+f1;
    }
  }

  for(k=0;k<fsize*fsize;k++){
    fil1[k] = fil1[k]/f2;
  }

  for(k=0;k<np*np;k++){
    fft_fil[k][0] = 0.0;
    fft_fil[k][1] = 0.0;
  }

  for(j=0;j<=nf;j++){
    for(i=0;i<=nf;i++){
      fft_fil[i+np*j][0] = fil1[(i+nf)+(j+nf)*fsize];
      fft_fil[i+np*j][1] = 0.0;
    }
    for(i=np-nf;i<np;i++){
      fft_fil[i+np*j][0] = fil1[(i+nf-np)+(j+nf)*fsize];
      fft_fil[i+np*j][1] = 0.0;
    }
  }

  for(j=np-nf;j<np;j++){
    for(i=0;i<=nf;i++){
      fft_fil[i+np*j][0] = fil1[(i+nf)+(j+nf-np)*fsize];
      fft_fil[i+np*j][1] = 0.0;
    }
    for(i=np-nf;i<np;i++){
      fft_fil[i+np*j][0] = fil1[(i+nf-np)+(j+nf-np)*fsize];
      fft_fil[i+np*j][1] = 0.0;
    }
  }

  delete[] fil1;

  fft_p = fftw_plan_dft_2d(np, np, fft_fil, fft_fil, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(fft_p);
  fftw_destroy_plan(fft_p);


  // normalize
  f1 = 1.0/((double)(np*np));
  for(k=0;k<np*np;k++){
    fft_fil[k][0] = fft_fil[k][0]*f1;
    fft_fil[k][1] = fft_fil[k][1]*f1;
  }

  // FFT data field
  for(j=0;j<np;j++){
    for(i=0;i<np;i++){
      if((i<ng) && (j<ng)){
        fft_gam[i+np*j][0] = in[i+ng*j];
        fft_gam[i+np*j][1] = 0.0;
      }
      else{
        fft_gam[i+np*j][0] = 0.0;
        fft_gam[i+np*j][1] = 0.0;
      }
    }
  }

  fft_p = fftw_plan_dft_2d(np, np, fft_gam, fft_gam, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(fft_p);
  fftw_destroy_plan(fft_p);

  // convolve with filter in fourier space
  for(k=0;k<np*np;k++){
    re = fft_gam[k][0]*fft_fil[k][0]-fft_gam[k][1]*fft_fil[k][1];
    im = fft_gam[k][0]*fft_fil[k][1]+fft_gam[k][1]*fft_fil[k][0];
    fft_gam[k][0] = re;
    fft_gam[k][1] = im;
  }

  fftw_free(fft_fil);

  fft_p = fftw_plan_dft_2d(np, np, fft_gam, fft_gam, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(fft_p);
  fftw_destroy_plan(fft_p);


  for(j=0;j<ng;j++){
    for(i=0;i<ng;i++){
      out[i+j*ng] = fft_gam[i+j*np][0];
    }
  }

  fftw_free(fft_gam);

  return;
}

void find_peaks_minima(double *kappa, double *gamma, int ng, çt offset, double *peaks, double *minima,
                       double numin, double numax, int Nnu){
  int vind;
  double v, dnu;
  bool flag_p, flag_m;


  dnu = (numax-numin)/((double) Nnu);

  for(int i=0;i<Nnu;++i){
    peaks[i] = 0.0;
    minima[i] = 0.0;
  }

  // search for peaks and minima
  for(int i=offset;i<ng-offset;++i){
    for(int j=offset;j<ng-offset;++j){
      v = kappa[j+ng*i];
      flag_p = true;
      flag_m = true;
      for(int di=-1;di<=1;++di){
        for(int dj=-1;dj<=1;++dj){
          if(di == 0 && dj == 0) continue;
          if(v < kappa[(j+dj)+ng*(i+di)]) flag_p = false;
          if(v > kappa[(j+dj)+ng*(i+di)]) flag_m = false;
        }
      }

      if(flag_p || flag_m){
        vind = (int) floor((v-numin)/dnu);
        if(vind >= 0 && vind < Nnu){
          if(flag_p) peaks[vind] += 1.0;
          if(flag_m) minima[vind] += 1.0;
        }
      }
    }
  }


  for(int i=0;i<Nnu;++i){
    peaks[i] = peaks[i]/dnu;
    minima[i] = minima[i]/dnu;
  }


  return;
}

void calc_PDF(double *kappa, double *gamma, int ng, int offset, double *PDF, double numin, double numax, int Nnu){
  double v, dnu, mu, norm;
  int vind;


  dnu = (numax-numin)/((double) Nnu);

  for(int i=0;i<Nnu;++i){
    PDF[i] = 0.0;
  }

  for(int i=offset;i<ng-offset;++i){
    for(int j=offset;j<ng-offset;++j){
      v = kappa[j+ng*i];
      mu = 1.0/(sqr(1.0-kappa[j+ng*i]) - sqr(gamma[j+ng*i]));
      vind = (int) floor((v-numin)/dnu);
      if(vind >= 0 && vind < Nnu){
        PDF[vind] += 1.0/mu;
      }
    }
  }


  norm = 0.0;
  for(int i=0;i<Nnu;++i){
    norm += PDF[i];
  }

  for(int i=0;i<Nnu;++i){
    PDF[i] = PDF[i]/norm/dnu;
  }


  return;
}

double calc_kappa2(double *kappa, double *gamma, int ng){
  double res, norm, mu;


  res = 0.0;
  norm = 0.0;

  for(int i=0;i<ng;++i){
    for(int j=0;j<ng;++j){
      mu = 1.0/(sqr(1.0-kappa[j+ng*i]) - sqr(gamma[j+ng*i]));
      res += sqr(kappa[j+ng*i])/mu;
      norm += 1.0/mu;
    }
  }

  res = res/norm;

  return res;
}
