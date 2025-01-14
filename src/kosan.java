/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author acer
 */
public class kosan {
    private String nama, jenis, harga;
    private int kontrak, kembali, total, harga2, uangku;
    private long nohp;
    
    public kosan() {
        nama = "";
        jenis = "";
        harga = "";
        kontrak = 0;
        nohp = 0;
        kembali = 0;
        total = 0;
        harga2 = 0;
        uangku = 0;
    }

    public int getUangku() {
        return uangku;
    }

    public void setUangku(int uangku) {
        this.uangku = uangku;
    }


    public int getHarga2() {
        return harga2;
    }

    public void setHarga2(int harga2) {
        this.harga2 = harga2;
    }

    public String getNama() {
        return nama;
    }

    public void setNama(String nama) {
        this.nama = nama;
    }

    public String getJenis() {
        return jenis;
    }

    public void setJenis(String jenis) {
        this.jenis = jenis;
    }

    public String getHarga() {
        return harga;
    }

    public void setHarga(String harga) {
        this.harga = harga;
    }

    public int getKontrak() {
        return kontrak;
    }

    public void setKontrak(int durasi) {
        this.kontrak = durasi;
    }

    public long getNohp() {
        return nohp;
    }

    public void setNohp(long nohp) {
        this.nohp = nohp;
    }

    public int getKembali() {
        return kembali;
    }

    public void setKembali(int kembali) {
        this.kembali = kembali;
    }

    public int getTotal() {
        return total;
    }

    public void setTotal(int total) {
        this.total = total;
    }
    
}
